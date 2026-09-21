import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .backend import MODEL, BackendError
from .schemas import EvaluationRequest, EvaluationResponse
from .scoring import answer, probabilities

ROOT = Path(__file__).parent


def create_app(backend, temperature=1.0):
    model_id = getattr(backend, "model_id", MODEL)
    probability_source = getattr(backend, "probability_source", "masked_position_logits")

    @asynccontextmanager
    async def lifespan(app):
        yield
        await backend.close()

    app = FastAPI(title="Diffusion Jev", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        ready = await backend.ready()
        return {
            "ready": ready,
            "model": model_id,
            "display_name": getattr(backend, "display_name", "LLaDA 2.1 mini"),
            "supports_images": getattr(backend, "supports_images", False),
            "backend": "sglang",
            "probability_source": probability_source,
            "temperature": temperature,
        }

    @app.get("/api/examples")
    async def examples():
        return json.loads((ROOT / "data/presets.json").read_text())

    @app.get("/api/datasets/{dataset_id}")
    async def dataset(dataset_id: str):
        from .vision import dataset_manifest

        try:
            data = dataset_manifest(dataset_id)
            return {k: v for k, v in data.items() if k != "images"}
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(
                404, "Dataset unavailable; run its preparation script in scripts/"
            ) from error

    @app.get("/api/datasets/{dataset_id}/images")
    async def gallery(
        dataset_id: str,
        split: str = "test",
        label: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
    ):
        from .vision import dataset_manifest

        try:
            data = dataset_manifest(dataset_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(404, "Image dataset is not prepared") from error
        selected = [
            row
            for row in data["images"]
            if (split == "all" or row["split"] == split) and (not label or row["label"] == label)
        ]
        return {
            "total": len(selected),
            "items": [
                {k: row[k] for k in ("id", "label", "split", "width", "height", "benchmark")}
                for row in selected[offset : offset + limit]
            ],
        }

    @app.get("/api/datasets/{dataset_id}/image/{image_id}")
    async def gallery_image(dataset_id: str, image_id: str):
        from .vision import dataset_image_path

        try:
            return FileResponse(dataset_image_path(dataset_id, image_id), media_type="image/jpeg")
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/v1/systemone", response_model=EvaluationResponse)
    async def evaluate(request: EvaluationRequest):
        if request.model not in ("jev", "diffusion-jev", model_id):
            raise HTTPException(422, f"Unknown model; use jev, diffusion-jev or {model_id}")
        started = time.perf_counter()
        try:
            if request.images and not getattr(backend, "supports_images", False):
                raise ValueError("The active model does not support images")
            if request.images:
                from .vision import resolve_images

                images = await asyncio.to_thread(resolve_images, request.images)
            else:
                images = []
            values = await asyncio.gather(
                *(
                    backend.score(request.state, q, images=images)
                    if images
                    else backend.score(request.state, q)
                    for q in request.questions.values()
                )
            )
            answers = {
                key: answer(q, probabilities(value["logits"], temperature))
                for (key, q), value in zip(request.questions.items(), values, strict=True)
            }
        except BackendError as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {
            "model": model_id,
            "answers": answers,
            "usage": {
                "input_tokens": sum(v["input_tokens"] for v in values),
                "output_tokens": sum(v["output_tokens"] for v in values),
            },
            "meta": {
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "probability_source": probability_source,
                "temperature": temperature,
                "passes": values[0]["passes"],
                "candidate_logits": {
                    key: v["logits"] for key, v in zip(request.questions, values, strict=True)
                },
                **{
                    field: {key: v[field] for key, v in zip(request.questions, values, strict=True)}
                    for field in (
                        "denoising_steps",
                        "candidate_mass",
                        "unrestricted_first_token",
                        "answer_position",
                    )
                    if all(field in v for v in values)
                },
            },
        }

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="playground")
    return app
