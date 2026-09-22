import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from diffusion_jev.cloud import DemoLimit, Jobs, Quota, create_cloud_app


class Store:
    def __init__(self):
        self.data = {}

    async def put_if_absent(self, key, value):
        if key in self.data:
            return False
        self.data[key] = value
        return True

    async def put(self, key, value):
        self.data[key] = value

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)


class Worker:
    def __init__(self):
        self.calls = []

    async def submit(self, payload):
        self.calls.append(payload)
        return "call-1"

    async def result(self, call_id):
        return {"status": "pending"}


PAYLOAD = {
    "model": "diffusion-jev",
    "state": "hello",
    "questions": {"q": {"type": "noul", "instructions": "Is this a greeting?"}},
}


def test_browsing_and_polling_cannot_wake_gpu():
    worker = Worker()
    with TestClient(create_cloud_app(Jobs(Store(), worker))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["execution"] == "queued"
        assert client.get("/api/examples").status_code == 200
        assert client.get("/api/jobs/" + str(uuid4())).status_code == 404
        assert client.post("/v1/systemone", json=PAYLOAD).status_code == 409
    assert worker.calls == []


def test_retries_submit_once_and_validate_before_spending():
    worker = Worker()
    with TestClient(create_cloud_app(Jobs(Store(), worker))) as client:
        headers = {"x-request-id": str(uuid4())}
        for _ in range(2):
            result = client.post("/api/jobs", json=PAYLOAD, headers=headers)
            assert result.status_code == 202
        assert len(worker.calls) == 1
        assert client.get("/api/jobs/" + result.json()["job_id"]).json()["status"] == "pending"
        changed = {**PAYLOAD, "state": "different"}
        assert client.post("/api/jobs", json=changed, headers=headers).status_code == 409
        invalid = {**PAYLOAD, "images": ["data:image/png;base64,bad!"]}
        assert (
            client.post(
                "/api/jobs", json=invalid, headers={"x-request-id": str(uuid4())}
            ).status_code
            == 422
        )
        oversized = {**PAYLOAD, "questions": {str(i): PAYLOAD["questions"]["q"] for i in range(5)}}
        assert client.post("/api/jobs", json=oversized, headers=headers).status_code == 422
        assert len(worker.calls) == 1


@pytest.mark.asyncio
async def test_quota_is_shared_atomic_and_survives_restarts():
    store = Store()
    now = datetime(2026, 9, 22, tzinfo=UTC)
    results = await asyncio.gather(
        *(Quota(store, monthly=2, daily=2).reserve(now) for _ in range(8)),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 2
    assert all(result is None or isinstance(result, DemoLimit) for result in results)
    with pytest.raises(DemoLimit):
        await Quota(store, monthly=2, daily=2).reserve(datetime(2026, 9, 23, tzinfo=UTC))
    await Quota(store, monthly=2, daily=2).reserve(datetime(2026, 10, 1, tzinfo=UTC))


def test_exhausted_allowance_does_not_start_another_gpu_job():
    worker = Worker()
    with TestClient(create_cloud_app(Jobs(Store(), worker, daily=1))) as client:
        assert (
            client.post(
                "/api/jobs", json=PAYLOAD, headers={"x-request-id": str(uuid4())}
            ).status_code
            == 202
        )
        response = client.post("/api/jobs", json=PAYLOAD, headers={"x-request-id": str(uuid4())})
        assert response.status_code == 429
        assert "allowance" in response.json()["detail"]
        assert len(worker.calls) == 1
