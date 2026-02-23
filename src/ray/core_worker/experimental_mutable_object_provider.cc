// Copyright 2024 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/core_worker/experimental_mutable_object_provider.h"

#include <memory>
#include <utility>
#include <vector>

namespace ray {
namespace core {
namespace experimental {

MutableObjectProvider::MutableObjectProvider(
    std::shared_ptr<plasma::PlasmaClientInterface> plasma,
    RayletFactory raylet_client_factory,
    std::function<Status(void)> check_signals)
    : plasma_(std::move(plasma)),
      object_manager_(std::make_shared<ray::experimental::MutableObjectManager>(
          std::move(check_signals))),
      raylet_client_factory_(std::move(raylet_client_factory)) {}

MutableObjectProvider::~MutableObjectProvider() {
  for (std::unique_ptr<boost::asio::executor_work_guard<
           boost::asio::io_context::executor_type>> &io_work : io_works_) {
    io_work->reset();
  }
  RAY_CHECK(object_manager_->SetErrorAll().code() == StatusCode::OK);

  for (std::unique_ptr<std::thread> &io_thread : io_threads_) {
    RAY_CHECK(io_thread->joinable());
    io_thread->join();
  }
}

void MutableObjectProvider::RegisterWriterChannel(
    const ObjectID &writer_object_id, const std::vector<NodeID> &remote_reader_node_ids) {
  {
    std::unique_ptr<plasma::MutableObject> writer_object;
    RAY_CHECK_OK(plasma_->GetExperimentalMutableObject(writer_object_id, &writer_object));
    RAY_CHECK_OK(object_manager_->RegisterChannel(
        writer_object_id, std::move(writer_object), /*reader=*/false));
    // `object` is now a nullptr.
  }

  if (remote_reader_node_ids.empty()) {
    return;
  }

  std::shared_ptr<std::vector<std::shared_ptr<RayletClientInterface>>> remote_readers =
      std::make_shared<std::vector<std::shared_ptr<RayletClientInterface>>>();
  // TODO(sang): Currently, these attributes are not cleaned up.
  // Start a thread that repeatedly listens for values on this object and then sends
  // them via RPC to the remote reader.
  io_contexts_.push_back(std::make_unique<instrumented_io_context>());
  instrumented_io_context &io_context = *io_contexts_.back();
  io_works_.push_back(
      std::make_unique<
          boost::asio::executor_work_guard<boost::asio::io_context::executor_type>>(
          io_context.get_executor()));

  // Find remote readers.
  for (const auto &node_id : remote_reader_node_ids) {
    // NOTE: Not setting local address because we're not testing compiled graphs with
    // testing_rpc_failure_avoid_intra_node_failures for now.
    client_call_managers_.push_back(std::make_unique<rpc::ClientCallManager>(
        io_context, /*record_stats=*/false, /*local_address=*/"always not local"));
    std::shared_ptr<RayletClientInterface> reader = raylet_client_factory_(node_id);
    remote_readers->push_back(reader);
  }

  // TODO(jhumphri): Extend this to support multiple channels. Currently, we must have
  // one thread per channel because the thread blocks on the channel semaphore.
  // TODO(sang): We currently create a thread per object id. It is not scalable.
  // We should instead just use a pool of threads.
  io_context.post(
      [this,
       &io_context,
       writer_object_id,
       remote_readers = std::move(remote_readers)]() {
        PollWriterClosure(io_context, writer_object_id, remote_readers);
      },
      "experimental::MutableObjectProvider.PollWriter");
  io_threads_.push_back(std::make_unique<std::thread>(
      &MutableObjectProvider::RunIOContext, this, std::ref(io_context)));
}

void MutableObjectProvider::RegisterReaderChannel(const ObjectID &object_id) {
  std::unique_ptr<plasma::MutableObject> object;
  RAY_CHECK_OK(plasma_->GetExperimentalMutableObject(object_id, &object));
  RAY_CHECK_OK(
      object_manager_->RegisterChannel(object_id, std::move(object), /*reader=*/true));
  // `object` is now a nullptr.
}

void MutableObjectProvider::HandleRegisterMutableObject(
    const ObjectID &writer_object_id,
    int64_t num_readers,
    const ObjectID &reader_object_id) {
  absl::MutexLock guard(&remote_writer_object_to_local_reader_lock_);

  LocalReaderInfo info;
  info.num_readers = num_readers;
  info.local_object_id = reader_object_id;
  bool success =
      remote_writer_object_to_local_reader_.insert({writer_object_id, info}).second;
  RAY_CHECK(success);

  RegisterReaderChannel(reader_object_id);
}

void MutableObjectProvider::HandlePushMutableObject(
    const rpc::PushMutableObjectRequest &request, rpc::PushMutableObjectReply *reply) {
  LocalReaderInfo info;
  const ObjectID writer_object_id = ObjectID::FromBinary(request.writer_object_id());
  {
    absl::MutexLock guard(&remote_writer_object_to_local_reader_lock_);
    auto it = remote_writer_object_to_local_reader_.find(writer_object_id);
    RAY_CHECK(it != remote_writer_object_to_local_reader_.end());
    info = it->second;
  }
  size_t total_data_size = request.total_data_size();
  size_t total_metadata_size = request.total_metadata_size();

  uint64_t offset = request.offset();
  uint64_t chunk_size = request.chunk_size();

  // Validate request bounds to prevent buffer overflows.
  RAY_CHECK_LE(offset + chunk_size, total_data_size)
      << "Chunk extends beyond total data size. offset=" << offset
      << ", chunk_size=" << chunk_size << ", total_data_size=" << total_data_size;
  RAY_CHECK_EQ(request.data().size(), chunk_size)
      << "Data size mismatch. Expected " << chunk_size << " bytes, got "
      << request.data().size() << " bytes";
  RAY_CHECK_EQ(request.metadata().size(), total_metadata_size)
      << "Metadata size mismatch. Expected " << total_metadata_size << " bytes, got "
      << request.metadata().size() << " bytes";

  // Simplified fault tolerance: whole-object retry with version-based stale detection.
  // The sender (PollWriterClosure) sends one version at a time sequentially.
  // If any chunk fails, the sender retries all chunks for that version.
  // No per-chunk dedup is needed since each chunk is sent exactly once per attempt.
  int64_t request_version = request.version();

  bool needs_write_acquire = false;
  {
    absl::MutexLock guard(&written_so_far_lock_);
    int64_t highest_completed = highest_completed_version_[writer_object_id];  // default 0

    if (request_version <= highest_completed) {
      // Stale retry from already-completed write.
      reply->set_done(true);
      return;
    }

    if (request_version > highest_completed + 1) {
      // Future version — should not happen with sequential sender. Reject defensively.
      reply->set_done(false);
      return;
    }

    // Active version (request_version == highest_completed + 1).
    if (!write_acquired_[writer_object_id]) {
      needs_write_acquire = true;
    }
  }

  // WriteAcquire and write logic for active version.
  bool object_complete = false;

  std::shared_ptr<Buffer> object_backing_store;
  if (needs_write_acquire) {
    {
      absl::MutexLock guard(&written_so_far_lock_);
      written_so_far_[writer_object_id] = 0;
    }
    // First chunk to arrive — acquire write lock and allocate backing store.
    RAY_CHECK_OK(object_manager_->WriteAcquire(info.local_object_id,
                                               total_data_size,
                                               /*metadata=*/nullptr,
                                               total_metadata_size,
                                               info.num_readers,
                                               object_backing_store));
    {
      absl::MutexLock guard(&written_so_far_lock_);
      write_acquired_[writer_object_id] = true;
    }
  } else {
    // Wait until WriteAcquire has completed before calling GetObjectBackingStore.
    {
      absl::MutexLock guard(&written_so_far_lock_);
      auto condition = [this, &writer_object_id]()
                           ABSL_SHARED_LOCKS_REQUIRED(written_so_far_lock_) {
                             return write_acquired_[writer_object_id];
                           };
      written_so_far_lock_.Await(absl::Condition(&condition));
    }
    RAY_CHECK_OK(object_manager_->GetObjectBackingStore(info.local_object_id,
                                                        total_data_size,
                                                        total_metadata_size,
                                                        object_backing_store));
  }
  RAY_CHECK(object_backing_store);

  // Copy chunk data to backing store.
  memcpy(object_backing_store->Data() + offset, request.data().data(), chunk_size);

  // Update progress. Use received_offsets_ to avoid double-counting when a
  // whole-object retry resends chunks that the receiver already has.
  {
    absl::MutexLock guard(&written_so_far_lock_);
    bool is_new_chunk = received_offsets_[writer_object_id].insert(offset).second;
    if (is_new_chunk) {
      written_so_far_[writer_object_id] += chunk_size;
      RAY_CHECK_LE(written_so_far_[writer_object_id], total_data_size);
    }
    if (written_so_far_[writer_object_id] == total_data_size) {
      object_complete = true;
    }
  }

  if (object_complete) {
    // All data chunks received — copy metadata and release write lock.
    memcpy(object_backing_store->Data() + total_data_size,
           request.metadata().data(),
           total_metadata_size);
    RAY_CHECK_OK(object_manager_->WriteRelease(info.local_object_id));

    {
      absl::MutexLock guard(&written_so_far_lock_);
      highest_completed_version_[writer_object_id] = request_version;
      written_so_far_.erase(writer_object_id);
      write_acquired_.erase(writer_object_id);
      received_offsets_.erase(writer_object_id);
    }

    reply->set_done(true);
  } else {
    reply->set_done(false);
  }
}

Status MutableObjectProvider::WriteAcquire(const ObjectID &object_id,
                                           int64_t data_size,
                                           const uint8_t *metadata,
                                           int64_t metadata_size,
                                           int64_t num_readers,
                                           std::shared_ptr<Buffer> &data,
                                           int64_t timeout_ms) {
  return object_manager_->WriteAcquire(
      object_id, data_size, metadata, metadata_size, num_readers, data, timeout_ms);
}

Status MutableObjectProvider::WriteRelease(const ObjectID &object_id) {
  return object_manager_->WriteRelease(object_id);
}

Status MutableObjectProvider::ReadAcquire(const ObjectID &object_id,
                                          std::shared_ptr<RayObject> &result,
                                          int64_t timeout_ms) {
  return object_manager_->ReadAcquire(object_id, result, timeout_ms);
}

Status MutableObjectProvider::ReadRelease(const ObjectID &object_id) {
  return object_manager_->ReadRelease(object_id);
}

Status MutableObjectProvider::SetError(const ObjectID &object_id) {
  return object_manager_->SetError(object_id);
}

Status MutableObjectProvider::GetChannelStatus(const ObjectID &object_id,
                                               bool is_reader) {
  return object_manager_->GetChannelStatus(object_id, is_reader);
}

void MutableObjectProvider::PushToReaderWithRetry(
    instrumented_io_context &io_context,
    const std::shared_ptr<RayletClientInterface> &reader,
    const ObjectID &writer_object_id,
    const std::shared_ptr<RayObject> &object,
    int64_t version,
    int retries_remaining,
    std::shared_ptr<size_t> num_replied,
    size_t total_readers,
    const std::shared_ptr<std::vector<std::shared_ptr<RayletClientInterface>>>
        &remote_readers,
    std::function<void()> on_all_done) {
  reader->PushMutableObject(
      writer_object_id,
      object->GetData()->Size(),
      object->GetMetadata()->Size(),
      object->GetData()->Data(),
      object->GetMetadata()->Data(),
      version,
      [this,
       &io_context,
       reader,
       writer_object_id,
       object,
       version,
       retries_remaining,
       num_replied,
       total_readers,
       remote_readers,
       on_all_done](const Status &push_object_status,
                     const rpc::PushMutableObjectReply &reply) {
        if (!push_object_status.ok()) {
          if (retries_remaining > 0) {
            RAY_LOG(WARNING)
                << "Failed to push mutable object " << writer_object_id
                << " version " << version << " to reader, retrying ("
                << retries_remaining << " retries remaining): " << push_object_status;
            // Schedule retry with backoff on the io_context thread.
            auto retry_timer = std::make_shared<boost::asio::steady_timer>(
                io_context,
                std::chrono::milliseconds(100 * (kMaxPushRetries - retries_remaining + 1)));
            retry_timer->async_wait(
                [this,
                 retry_timer,
                 &io_context,
                 reader,
                 writer_object_id,
                 object,
                 version,
                 retries_remaining,
                 num_replied,
                 total_readers,
                 remote_readers,
                 on_all_done](const boost::system::error_code &ec) {
                  if (ec) {
                    // Timer cancelled (e.g., shutdown). Count as replied.
                    *num_replied += 1;
                    if (*num_replied == total_readers) {
                      on_all_done();
                    }
                    return;
                  }
                  PushToReaderWithRetry(io_context,
                                        reader,
                                        writer_object_id,
                                        object,
                                        version,
                                        retries_remaining - 1,
                                        num_replied,
                                        total_readers,
                                        remote_readers,
                                        on_all_done);
                });
          } else {
            RAY_LOG(ERROR)
                << "Failed to transfer mutable object " << writer_object_id
                << " version " << version
                << " to a remote reader after all retries. This may cause a hang.";
            *num_replied += 1;
            if (*num_replied == total_readers) {
              on_all_done();
            }
          }
        } else {
          *num_replied += 1;
          if (*num_replied == total_readers) {
            on_all_done();
          }
        }
      });
}

void MutableObjectProvider::PollWriterClosure(
    instrumented_io_context &io_context,
    const ObjectID &writer_object_id,
    const std::shared_ptr<std::vector<std::shared_ptr<RayletClientInterface>>>
        &remote_readers) {
  // NOTE: There's only 1 PollWriterClosure at any time in a single thread.
  std::shared_ptr<RayObject> object;
  int64_t version = 0;
  // The corresponding ReadRelease() will be automatically called when
  // `object` goes out of scope.
  Status status =
      object_manager_->ReadAcquire(writer_object_id, object, version, /*timeout_ms=*/-1);
  // Check if the thread returned from ReadAcquire() because the process is exiting, not
  // because there is something to read.
  if (status.code() == StatusCode::ChannelError) {
    // The process is exiting.
    return;
  }
  RAY_CHECK_EQ(static_cast<int>(status.code()), static_cast<int>(StatusCode::OK));

  RAY_CHECK(object->GetData());
  RAY_CHECK(object->GetMetadata());

  // Version was obtained safely from ReadAcquire (with header_sem protection)
  RAY_CHECK_GT(version, 0) << "Invalid version for " << writer_object_id;

  std::shared_ptr<size_t> num_replied = std::make_shared<size_t>(0);
  auto on_all_done = [this, &io_context, writer_object_id, remote_readers]() {
    io_context.post(
        [this, &io_context, writer_object_id, remote_readers]() {
          PollWriterClosure(io_context, writer_object_id, remote_readers);
        },
        "experimental::MutableObjectProvider.PollWriter");
  };
  for (const auto &reader : *remote_readers) {
    PushToReaderWithRetry(io_context,
                          reader,
                          writer_object_id,
                          object,
                          version,
                          kMaxPushRetries,
                          num_replied,
                          remote_readers->size(),
                          remote_readers,
                          on_all_done);
  }
}

void MutableObjectProvider::RunIOContext(instrumented_io_context &io_context) {
// TODO(jhumphri): Decompose this.
#ifndef _WIN32
  // Block SIGINT and SIGTERM so they will be handled by the main thread.
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &mask, nullptr);

#endif
  SetThreadName("worker.channel_io");
  io_context.run();
  RAY_LOG(INFO) << "Core worker channel io service stopped.";
}

}  // namespace experimental
}  // namespace core
}  // namespace ray
