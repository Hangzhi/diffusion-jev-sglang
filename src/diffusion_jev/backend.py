import asyncio
import math

import httpx

from .scoring import LABELS, messages, options

MODEL = "inclusionAI/LLaDA2.1-mini"
REVISION = "20e64e2ad21644d0e5248586ed9c942cdd45de0f"


class BackendError(RuntimeError):
    pass


class SGLangBackend:
    def __init__(self, url, tokenizer, concurrency=8, expected_passes=1):
        self.expected_passes = expected_passes
        self.url = url.rstrip("/")
        self.tokenizer = tokenizer
        self.client = httpx.AsyncClient(timeout=180)
        self.limit = asyncio.Semaphore(concurrency)
        self.candidate_ids = candidate_ids(tokenizer)

    async def close(self):
        await self.client.aclose()

    async def ready(self):
        try:
            response = await self.client.get(self.url + "/health", timeout=2)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def score(self, state, question):
        prompt = self.tokenizer.apply_chat_template(
            messages(state, question), tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if 156895 in prompt:
            raise ValueError("State/instructions cannot contain the model mask token")
        if len(prompt) > 7900:
            raise ValueError("Question exceeds the 7,900-token prompt limit")
        async with self.limit:
            try:
                response = await self.client.post(
                    self.url + "/generate",
                    json={
                        "input_ids": prompt,
                        "sampling_params": {
                            "max_new_tokens": 1,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("meta_info"), dict):
                    raise BackendError("Malformed SGLang metadata")
                meta = payload["meta_info"]
                vectors = meta.get("jev_candidate_logits")
                if (
                    not isinstance(vectors, list)
                    or len(vectors) != 1
                    or not isinstance(vectors[0], list)
                    or len(vectors[0]) != 26
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vectors[0])
                ):
                    raise BackendError(
                        "SGLang did not return the Jev scoring extension metadata. Start the patched engine with JevScoring."
                    )
                if meta.get("jev_passes") != [self.expected_passes]:
                    raise BackendError(
                        "Engine pass setting differs from API configuration; restart with matching --passes"
                    )
                return {
                    "logits": vectors[0][: len(options(question))],
                    "input_tokens": meta["prompt_tokens"],
                    "output_tokens": meta["completion_tokens"],
                    "passes": meta["jev_passes"][0],
                }
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                raise BackendError("SGLang inference failed; inspect the engine log") from error


def candidate_ids(tokenizer):
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(item) != 1 for item in ids) or len({item[0] for item in ids}) != 26:
        raise ValueError("Tokenizer must encode each A-Z label as a distinct single token")
    return [item[0] for item in ids]
