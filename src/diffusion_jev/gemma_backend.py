"""Typed decisions from the native DiffusionGemma denoiser in SGLang."""

import math

import httpx

from .backend import BackendError, SGLangBackend
from .scoring import messages, options

MODEL = "google/diffusiongemma-26B-A4B-it"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
SGLANG_REVISION = "ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f"
# The checkpoint emits this empty thought block even with thinking disabled.
# Read the answer after this generated prefix inside the native diffusion canvas.
ANSWER_PREFIX = "<|channel>thought\n<channel|>"


class DiffusionGemmaBackend(SGLangBackend):
    model_id = MODEL
    display_name = "DiffusionGemma 26B A4B"
    supports_images = True
    probability_source = "self_conditioned_denoiser_logits"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_prefix_ids = self.tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)

    async def score(self, state, question, images=None):
        chat = messages(state, question)
        if images:
            chat[1]["content"] = [
                *({"type": "image"} for _ in images),
                {"type": "text", "text": chat[1]["content"]},
            ]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(self.tokenizer.encode(prompt, add_special_tokens=False)) > 7000:
            raise ValueError("Question exceeds the 7,000-token text prompt limit")
        request = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": len(self.answer_prefix_ids) + 1,
                "temperature": 0,
            },
        }
        if images:
            request["image_data"] = images
        async with self.limit:
            try:
                response = await self.client.post(self.url + "/generate", json=request)
                response.raise_for_status()
                payload = response.json()
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
                        "DiffusionGemma denoiser readout metadata is missing or invalid"
                    )
                if meta.get("jev_max_denoising_steps") != [self.expected_passes]:
                    raise BackendError(
                        "DiffusionGemma denoising configuration does not match the API"
                    )
                if meta.get("jev_answer_prefix_tokens") != [self.answer_prefix_ids]:
                    raise BackendError(
                        "DiffusionGemma did not produce the expected empty thought prefix"
                    )
                for field in (
                    "jev_denoising_steps",
                    "jev_candidate_mass",
                    "jev_unrestricted_token",
                ):
                    if not isinstance(meta.get(field), list) or len(meta[field]) != 1:
                        raise BackendError("DiffusionGemma readout diagnostics are invalid")
                steps = meta["jev_denoising_steps"][0]
                raw_mass = meta["jev_candidate_mass"][0]
                token = meta["jev_unrestricted_token"][0]
                if (
                    type(steps) is not int
                    or not 1 <= steps <= self.expected_passes
                    or type(raw_mass) not in (int, float)
                    or not math.isfinite(raw_mass)
                    or not 0 <= raw_mass <= 1.0001
                    or type(token) is not int
                    or token < 0
                ):
                    raise BackendError("DiffusionGemma readout diagnostics are invalid")
                logits = vectors[0]
                count = len(options(question))
                relative = [math.exp(value - max(logits)) for value in logits]
                mass = raw_mass * sum(relative[:count]) / sum(relative)
                return {
                    "logits": logits[:count],
                    "input_tokens": meta["prompt_tokens"],
                    "output_tokens": meta["completion_tokens"],
                    "passes": self.expected_passes,
                    "denoising_steps": steps,
                    "candidate_mass": min(1.0, max(0.0, mass)),
                    "unrestricted_first_token": self.tokenizer.decode([token]),
                    "answer_position": len(self.answer_prefix_ids),
                }
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                raise BackendError(
                    "DiffusionGemma inference failed; inspect the engine log"
                ) from error
