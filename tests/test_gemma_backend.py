import asyncio
import json

import httpx
import pytest

from diffusion_jev.backend import BackendError
from diffusion_jev.gemma_backend import ANSWER_PREFIX, DiffusionGemmaBackend
from diffusion_jev.schemas import Choice


class Tokenizer:
    def encode(self, text, **kwargs):
        if text == ANSWER_PREFIX:
            return [100, 45518, 107, 101]
        return [ord(text)] if len(text) == 1 else [1] * 100

    def decode(self, ids):
        return chr(ids[0])

    def apply_chat_template(self, chat, **kwargs):
        self.chat = chat
        return "formatted prompt"


def metadata():
    return {
        "jev_candidate_logits": [[float(i) for i in range(26)]],
        "jev_max_denoising_steps": [48],
        "jev_denoising_steps": [7],
        "jev_candidate_mass": [0.8],
        "jev_unrestricted_token": [65],
        "jev_answer_prefix_tokens": [[100, 45518, 107, 101]],
        "prompt_tokens": 400,
        "completion_tokens": 1,
    }


def run(meta):
    async def evaluate():
        tokenizer = Tokenizer()
        backend = DiffusionGemmaBackend("http://engine", tokenizer, expected_passes=48)
        await backend.client.aclose()
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"meta_info": meta})

        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            result = await backend.score(
                "Classify the image.",
                Choice(
                    type="choice",
                    instructions="Which flower?",
                    criteria={"rose": "Rose", "tulip": "Tulip"},
                ),
                images=["data:image/jpeg;base64,AAAA"],
            )
            return result, tokenizer.chat, requests[0]
        finally:
            await backend.close()

    return asyncio.run(evaluate())


def test_image_request_and_candidate_mass_use_actual_offered_options():
    result, chat, request = run(metadata())
    assert chat[1]["content"][0] == {"type": "image"}
    assert request["image_data"] == ["data:image/jpeg;base64,AAAA"]
    assert request["sampling_params"]["max_new_tokens"] == 5
    assert request["text"] == "formatted prompt"
    assert result["logits"] == [0.0, 1.0]
    assert result["denoising_steps"] == 7
    assert 0 < result["candidate_mass"] < 0.001


@pytest.mark.parametrize(
    "field,value",
    [
        ("jev_candidate_logits", []),
        ("jev_max_denoising_steps", [1]),
        ("jev_denoising_steps", [49]),
        ("jev_candidate_mass", []),
        ("jev_candidate_mass", [2.0]),
        ("jev_unrestricted_token", [-1]),
        ("jev_answer_prefix_tokens", [[100, 45518, 107, 0]]),
    ],
)
def test_invalid_readout_fails_without_fabricating_probabilities(field, value):
    data = metadata()
    data[field] = value
    with pytest.raises(BackendError):
        run(data)
