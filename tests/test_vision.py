import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from diffusion_jev import vision
from diffusion_jev.api import create_app
from diffusion_jev.schemas import EvaluationRequest


@pytest.fixture
def flowers(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "DATA_ROOT", tmp_path)
    vision.flower_manifest.cache_clear()
    image_id = "a" * 64
    directory = tmp_path / "flowers/images"
    directory.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(directory / f"{image_id}.jpg")
    (tmp_path / "flowers/manifest.json").write_text(
        json.dumps(
            {
                "count": 1,
                "labels": ["rose", "tulip"],
                "images": [
                    {
                        "id": image_id,
                        "label": "rose",
                        "split": "test",
                        "width": 24,
                        "height": 16,
                        "benchmark": True,
                        "original_member": "rose/answer-is-rose.jpg",
                    },
                ],
            }
        )
    )
    yield image_id
    vision.flower_manifest.cache_clear()


class VisionBackend:
    model_id = "google/diffusiongemma-26B-A4B-it"
    supports_images = True
    probability_source = "self_conditioned_denoiser_logits"

    async def ready(self):
        return True

    async def close(self):
        pass

    async def score(self, state, question, images=None):
        self.received = (state, images)
        return {
            "logits": [1.0, 0.0],
            "input_tokens": 300,
            "output_tokens": 1,
            "passes": 48,
            "denoising_steps": 12,
            "candidate_mass": 0.85,
            "unrestricted_first_token": "A",
        }


def test_gallery_sends_pixels_without_filename_or_ground_truth(flowers):
    backend = VisionBackend()
    with TestClient(create_app(backend)) as client:
        listing = client.get("/api/datasets/flowers/images").json()
        assert listing["total"] == 1
        assert "original_member" not in listing["items"][0]
        assert client.get("/api/datasets/flowers/images?label=tulip").json()["total"] == 0
        assert (
            client.get(f"/api/datasets/flowers/image/{flowers}").headers["content-type"]
            == "image/jpeg"
        )
        result = client.post(
            "/v1/systemone",
            json={
                "model": "jev",
                "state": "Classify this image.",
                "images": [flowers],
                "questions": {
                    "flower": {
                        "type": "choice",
                        "instructions": "Which flower?",
                        "criteria": {"rose": "Rose", "tulip": "Tulip"},
                    }
                },
            },
        )
        assert result.status_code == 200, result.text
        assert result.json()["model"] == backend.model_id
        assert result.json()["meta"]["denoising_steps"] == {"flower": 12}
        assert backend.received[0] == "Classify this image."
        data_url = backend.received[1][0]
        assert data_url.startswith("data:image/jpeg;base64,")
        assert flowers not in data_url and "rose" not in data_url
        decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        assert decoded.size == (24, 16)


@pytest.mark.parametrize(
    "source",
    [
        "../../etc/passwd",
        "http://127.0.0.1/private",
        "file:///etc/passwd",
        "data:image/jpeg;base64,not-valid!",
    ],
)
def test_image_inputs_reject_paths_urls_and_invalid_base64(source):
    with pytest.raises(ValueError):
        vision.resolve_images([source])


def test_upload_normalizes_image_and_removes_metadata():
    raw = io.BytesIO()
    Image.new("RGB", (1500, 1000), "green").save(raw, format="PNG")
    normalized = vision.resolve_images(
        ["data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()]
    )[0]
    image = Image.open(io.BytesIO(base64.b64decode(normalized.split(",", 1)[1])))
    assert image.format == "JPEG" and image.size == (768, 512)


def test_images_do_not_bypass_text_request_bound():
    with pytest.raises(ValueError, match="100,000"):
        EvaluationRequest.model_validate(
            {
                "model": "jev",
                "state": "x" * 100_001,
                "images": ["a" * 64],
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            }
        )


def test_text_only_backend_rejects_visual_input(flowers):
    backend = VisionBackend()
    backend.supports_images = False
    with TestClient(create_app(backend)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "jev",
                "state": "state",
                "images": [flowers],
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
        assert response.status_code == 422
        assert not hasattr(backend, "received")


def test_multiple_galleries_are_isolated_and_labels_stay_out_of_inference(flowers):
    image_id = "b" * 64
    directory = vision.DATA_ROOT / "quickdraw/images"
    directory.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(directory / f"{image_id}.jpg")
    (directory.parent / "manifest.json").write_text(
        json.dumps(
            {
                "title": "Quick, Draw!",
                "count": 1,
                "labels": ["cat"],
                "images": [
                    {
                        "id": image_id,
                        "label": "cat",
                        "split": "demo",
                        "width": 32,
                        "height": 32,
                        "benchmark": False,
                        "source_key": "private-metadata",
                    }
                ],
            }
        )
    )
    backend = VisionBackend()
    with TestClient(create_app(backend)) as client:
        listing = client.get("/api/datasets/quickdraw/images?split=demo").json()
        assert listing["total"] == 1
        assert "source_key" not in listing["items"][0]
        assert client.get(f"/api/datasets/flowers/image/{image_id}").status_code == 404
        assert client.get("/api/datasets/unknown").status_code == 404
        result = client.post(
            "/v1/systemone",
            json={
                "model": "jev",
                "state": "Guess the sketch.",
                "images": [f"quickdraw:{image_id}"],
                "questions": {
                    "doodle": {
                        "type": "choice",
                        "instructions": "What is drawn?",
                        "criteria": {"cat": "Cat", "fish": "Fish"},
                    }
                },
            },
        )
        assert result.status_code == 200, result.text
        assert backend.received[0] == "Guess the sketch."
        data_url = backend.received[1][0]
        assert "quickdraw" not in data_url and image_id not in data_url
        assert Image.open(io.BytesIO(base64.b64decode(data_url.split(",")[1]))).size == (32, 32)
    with pytest.raises(ValueError):
        vision.resolve_images([f"unknown:{image_id}"])
    with pytest.raises(ValueError):
        vision.dataset_image_path("../quickdraw", image_id)
