from locust import HttpUser, TaskSet, between, task

from app.config import TEST_AI_SERVER_HOST


class UserAction(TaskSet):
    @task
    def get_integrate_predict(self):
        self.client.post(
            "/integrate_predict",
            json={
                "problem_id": 1234,
                "user_answer": "얄라얄라얄라숑 얄라리얄라 얄라얄라얄라숑 얄라리얄라 얄라얄라얄라숑 얄라리얄라 얄라얄라얄라숑 얄라리얄라 얄라얄라얄라숑 얄라리얄라 얄라얄라얄라숑 얄라리얄라",
                "keyword_standards": [
                    {"id": 96, "content": "keyword-1"},
                    {"id": 97, "content": "keyword-2"},
                    {"id": 98, "content": "keyword-3"},
                ],
                "content_standards": [
                    {"id": 123, "content": "content 1"},
                    {"id": 124, "content": "content 2"},
                    {"id": 125, "content": "content 3"},
                ],
            },
        )


class WebsiteUser(HttpUser):
    host = TEST_AI_SERVER_HOST
    tasks = [UserAction]
    wait_time = between(1, 4)
