import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .backend import MODEL, BackendError
from .schemas import EvaluationRequest, EvaluationResponse
from .scoring import answer, probabilities

ROOT = Path(__file__).parent


def create_app(backend, temperature=1.0):
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
            "model": MODEL,
            "backend": "sglang",
            "probability_source": "masked_position_logits",
            "temperature": temperature,
        }

    @app.get("/api/examples")
    async def examples():
        return json.loads((ROOT / "data/presets.json").read_text())

    @app.post("/v1/systemone", response_model=EvaluationResponse)
    async def evaluate(request: EvaluationRequest):
        if request.model not in ("diffusion-jev", MODEL):
            raise HTTPException(
                422, "Unknown model; use diffusion-jev or inclusionAI/LLaDA2.1-mini"
            )
        started = time.perf_counter()
        try:
            values = await asyncio.gather(
                *(backend.score(request.state, q) for q in request.questions.values())
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
            "model": MODEL,
            "answers": answers,
            "usage": {
                "input_tokens": sum(v["input_tokens"] for v in values),
                "output_tokens": sum(v["output_tokens"] for v in values),
            },
            "meta": {
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "probability_source": "masked_position_logits",
                "temperature": temperature,
                "passes": values[0]["passes"],
                "candidate_logits": {
                    key: v["logits"] for key, v in zip(request.questions, values, strict=True)
                },
            },
        }

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="playground")
    return app
