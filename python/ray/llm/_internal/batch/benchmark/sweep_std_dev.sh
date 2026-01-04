#!/bin/bash

# Script to sweep std-dev values and collect throughput results

# Default arguments
MODE="generate"
MODEL_NAME="Qwen/Qwen3-4B"
# MAX_DECODE_TOKENS=100
MAX_DECODE_TOKENS=(1 10 50 100 200)
IGNORE_EOS="--ignore-eos"
SYNC_ENGINE="--sync-engine"

# Array of std-dev values to sweep
# STD_DEVS=(0 10)
STD_DEV=0

# Results storage
declare -a RESULTS
declare -a THROUGHPUTS

echo "============================================================"
echo "Sweeping std-dev values: ${STD_DEVS[@]}"
echo "============================================================"
echo ""

# Loop through each std-dev value
for max_decode_tokens in "${MAX_DECODE_TOKENS[@]}"; do
    echo "Running benchmark with max-decode-tokens=${max_decode_tokens}..."
    echo "------------------------------------------------------------"
    
    # Run the benchmark and capture output
    OUTPUT=$(python benchmark_engine.py \
        --mode ${MODE} \
        --model-name ${MODEL_NAME} \
        --max-decode-tokens ${max_decode_tokens} \
        ${IGNORE_EOS} \
        --std-dev ${STD_DEV} 2>&1)
    
    # Extract throughput from output (looks for "Throughput  : XX.XX req/s")
    # Look for the line in the BENCHMARK summary section
    THROUGHPUT=$(echo "$OUTPUT" | grep -A 10 "BENCHMARK" | grep "Throughput" | sed -E 's/.*Throughput[[:space:]]+:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+req\/s.*/\1/' | head -1)
    
    if [ -z "$THROUGHPUT" ] || [ "$THROUGHPUT" = "" ]; then
        echo "ERROR: Could not extract throughput for std-dev=${std_dev}"
        echo "DEBUG: Searching for throughput line..."
        echo "$OUTPUT" | grep -A 10 "BENCHMARK" | grep "Throughput" || echo "No throughput line found"
        THROUGHPUT="N/A"
    else
        echo "✓ Throughput: ${THROUGHPUT} req/s"
    fi
    
    # Store results
    RESULTS+=("std-dev=${std_dev}: ${THROUGHPUT} req/s")
    THROUGHPUTS+=("${THROUGHPUT}")
    
    echo ""
done

# Print summary
echo "============================================================"
echo "SUMMARY - Throughput Results"
echo "============================================================"
for result in "${RESULTS[@]}"; do
    echo "$result"
done
echo "============================================================"

# Optional: Save to file
OUTPUT_FILE="std_dev_sweep_results_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Std-Dev Sweep Results"
    echo "Date: $(date)"
    echo "Model: ${MODEL_NAME}"
    echo "Mode: ${MODE}"
    echo "Max Decode Tokens: ${MAX_DECODE_TOKENS}"
    echo ""
    echo "Results:"
    for result in "${RESULTS[@]}"; do
        echo "$result"
    done
} > "$OUTPUT_FILE"

echo ""
echo "Results saved to: ${OUTPUT_FILE}"
