"""CPU-only public demo API. Reading pages or job status never invokes a GPU."""

import asyncio
import hashlib
import json
import re
import secrets
import time
from datetime import UTC, datetime

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .api import create_app
from .schemas import EvaluationRequest
from .vision import DATA_ROOT, resolve_images

MAX_BODY = 3_000_000


class DemoLimit(Exception):
    pass


class Quota:
    """Atomic slot reservations survive CPU restarts and concurrent deployments.

    Store.put_if_absent must be atomic (Modal Dict's skip_if_exists). Failed
    inference still consumes a slot: it can already have incurred startup cost.
    """

    def __init__(self, store, monthly=1000, daily=100):
        self.store, self.monthly, self.daily = store, monthly, daily

    async def claim_slot(self, window, limit):
        start = secrets.randbelow(limit)
        keys = [f"quota:{window}:{(start + offset) % limit}" for offset in range(limit)]
        if await self.store.put_if_absent(keys[0], True):
            return keys[0]
        # Batch reads near a full allowance, then atomically claim just one free slot.
        # Concurrent callers may see the same free slot; only one can reserve it.
        for offset in range(1, limit, 64):
            batch = keys[offset : offset + 64]
            occupied = await asyncio.gather(*(self.store.get(key) for key in batch))
            for key, value in zip(batch, occupied, strict=True):
                if not value and await self.store.put_if_absent(key, True):
                    return key
        return None

    async def reserve(self, now=None):
        now = now or datetime.now(UTC)
        acquired = []
        windows = [
            (now.strftime("%Y-%m-%dT%H:%M"), 10, "Please wait a minute before trying again."),
            (now.strftime("%Y-%m-%d"), self.daily, "Today's demo allowance is used up."),
            (now.strftime("%Y-%m"), self.monthly, "This month's demo allowance is used up."),
        ]
        for window, limit, message in windows:
            key = await self.claim_slot(window, limit)
            if key is None:
                for key in acquired:
                    await self.store.delete(key)
                raise DemoLimit(message + " You can still draw and browse the gallery.")
            acquired.append(key)


class Jobs:
    def __init__(self, store, worker, monthly=1000, daily=100):
        self.store, self.worker = store, worker
        self.quota = Quota(store, monthly, daily)

    async def submit(self, payload, request_id):
        if not re.fullmatch(r"[a-f0-9-]{36}", request_id):
            raise HTTPException(400, "A request ID is required.")
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = "job:" + request_id
        record = {"created": time.time(), "fingerprint": fingerprint, "status": "submitting"}
        if not await self.store.put_if_absent(key, record):
            previous = await self.store.get(key)
            if previous["fingerprint"] != fingerprint:
                raise HTTPException(
                    409, "This request ID was already used for a different drawing."
                )
            return request_id
        try:
            await self.quota.reserve()
            # Convert gallery IDs to pixels on the CPU, before submitting GPU work.
            if payload.get("images"):
                payload["images"] = resolve_images(payload["images"])
            call_id = await self.worker.submit(payload)
            await self.store.put(key, {**record, "status": "pending", "call_id": call_id})
        except DemoLimit as error:
            await self.store.delete(key)
            raise HTTPException(429, str(error)) from error
        except Exception:
            await self.store.put(
                key, {**record, "status": "failed", "detail": "Could not start the prediction."}
            )
            raise
        return request_id

    async def result(self, job_id):
        if not re.fullmatch(r"[a-f0-9-]{36}", job_id):
            raise HTTPException(404, "Prediction not found.")
        record = await self.store.get("job:" + job_id)
        if not record:
            raise HTTPException(404, "Prediction not found.")
        if record["status"] == "failed":
            return {"status": "failed", "detail": record["detail"]}
        if time.time() - record["created"] > 1200:
            return {"status": "failed", "detail": "This prediction expired. Please try again."}
        if not record.get("call_id"):
            return {"status": "pending"}
        return await self.worker.result(record["call_id"])


class _Capabilities:
    """Used only to reuse gallery routes; this object cannot perform inference."""

    async def close(self):
        pass


def create_cloud_app(jobs):
    app = create_app(_Capabilities())
    static = app.router.routes.pop()
    app.router.routes = [r for r in app.router.routes if r.path not in {"/health", "/v1/systemone"}]

    @app.get("/health")
    async def health():
        return {
            "ready": True,  # CPU admission is ready; this does not assert a warm GPU.
            "model": "google/diffusiongemma-26B-A4B-it",
            "display_name": "DiffusionGemma 26B A4B",
            "supports_images": True,
            "execution": "queued",
            "gpu_policy": "starts_on_request",
            "monthly_jobs": jobs.quota.monthly,
            "daily_jobs": jobs.quota.daily,
        }

    @app.post("/api/jobs", status_code=202)
    async def submit(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BODY:
                raise HTTPException(413, "The image is too large. Please use a smaller image.")
        try:
            data = EvaluationRequest.model_validate_json(raw)
        except ValidationError as error:
            raise HTTPException(422, "The prediction request is invalid.") from error
        if data.model not in {"jev", "diffusion-jev", "google/diffusiongemma-26B-A4B-it"}:
            raise HTTPException(422, "Unknown model.")
        if len(data.questions) > 4 or len(data.images) > 1:
            raise HTTPException(422, "The public demo allows one image and four questions per job.")
        if len(data.model_dump_json(exclude={"images"})) > 12_000:
            raise HTTPException(422, "Please use a shorter question or message.")
        payload = data.model_dump()
        try:
            # Reject invalid pixels before reserving quota or waking a worker.
            if data.images:
                resolve_images(data.images)
            job_id = await jobs.submit(payload, request.headers.get("x-request-id", ""))
        except ValueError as error:
            raise HTTPException(422, "Please choose a valid image.") from error
        return {"job_id": job_id, "status": "pending"}

    @app.get("/api/jobs/{job_id}")
    async def result(job_id: str):
        return JSONResponse(await jobs.result(job_id), headers={"Cache-Control": "no-store"})

    @app.post("/v1/systemone")
    async def synchronous_disabled():
        raise HTTPException(409, "This public demo uses POST /api/jobs and GET /api/jobs/{id}.")

    app.mount("/gallery", StaticFiles(directory=DATA_ROOT), name="cloud-gallery")
    app.router.routes.append(static)
    return app
