"""Simple Locust load test file for finding peak RPS."""
from locust import HttpUser, task, constant


class ServeUser(HttpUser):
    wait_time = constant(0)  # Very short wait time for high RPS

    @task
    def get_request(self):
        self.client.get("/")

