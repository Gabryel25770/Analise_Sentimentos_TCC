from locust import HttpUser, task, between

class SentimentUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def analyze_sentiment(self):
        self.client.post("/analyze", json={"text": "estou feliz hoje", "tipo": "geral"})
