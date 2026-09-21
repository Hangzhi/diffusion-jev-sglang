import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from diffusion_jev.api import create_app
from diffusion_jev.backend import BackendError
from diffusion_jev.engine_setup import ANCHOR, MARKER, patch
from diffusion_jev.evaluation import fit_calibration, metrics
from diffusion_jev.schemas import Choice, EvaluationRequest, Noul, Score
from diffusion_jev.scoring import answer, messages, probabilities


class Backend:
    async def ready(self):
        return True

    async def close(self):
        pass

    async def score(self, state, q):
        count = 2 if q.type == "noul" else len(q.criteria)
        return {"logits": list(range(count)), "input_tokens": 17, "output_tokens": 1, "passes": 1}


def test_softmax_and_typed_answers():
    assert probabilities([10000, 10000]) == [0.5, 0.5]
    assert probabilities([1, 2], 2)[1] < probabilities([1, 2])[1]
    assert answer(Noul(type="noul", instructions="?"), [0.2, 0.8]) == {"type": "noul", "noul": 0.8}
    result = answer(
        Score(type="score", instructions="?", criteria=["low", "middle", "high"]), [0.2, 0.3, 0.5]
    )
    assert result["score"] == 1.3
    assert result["legend"] == {"0": "low", "1": "middle", "2": "high"}
    assert 0 <= result["confidence"] <= 1
    result = answer(
        Choice(type="choice", instructions="?", criteria={"é": None, "B": None}), [0.5, 0.5]
    )
    assert result["choice"] == "é" and result["confidence"] == 0


@pytest.mark.parametrize(
    "values,t", [([0, float("nan")], 1), ([0, 1], 0), ([0, 1], float("inf")), ([1], 1)]
)
def test_invalid_probabilities(values, t):
    with pytest.raises(ValueError):
        probabilities(values, t)


def test_api_contract_all_types():
    request = {
        "model": "diffusion-jev",
        "state": {"message": "test"},
        "questions": {
            "bool": {"type": "noul", "instructions": ["Is it positive?"]},
            "choice": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": None, "b": {"detail": "b"}},
            },
            "score": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        },
    }
    with TestClient(create_app(Backend())) as client:
        response = client.post("/v1/systemone", json=request)
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["usage"] == {"input_tokens": 51, "output_tokens": 3}
        assert set(data["answers"]) == set(request["questions"])
        assert math.isclose(sum(data["answers"]["choice"]["probabilities"].values()), 1)
        request["model"] = "jev-latest"
        assert client.post("/v1/systemone", json=request).status_code == 422
        assert client.get("/health").json()["ready"]
        assert client.get("/api/examples").status_code == 200
        assert client.get("/").status_code == 200


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": {"only": None}},
        {"type": "choice", "instructions": "?", "criteria": dict.fromkeys(map(str, range(27)))},
        {"type": "score", "instructions": "?", "criteria": ["only"]},
        {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}},
        {"type": "unknown", "instructions": "?"},
    ],
)
def test_validation(question):
    with TestClient(create_app(Backend())) as c:
        assert (
            c.post(
                "/v1/systemone",
                json={"model": "diffusion-jev", "state": "x", "questions": {"q": question}},
            ).status_code
            == 422
        )


def test_backend_failure_is_not_fake_prediction():
    class Failed(Backend):
        async def score(self, *args):
            raise BackendError("No logits")

    with TestClient(create_app(Failed())) as c:
        r = c.post(
            "/v1/systemone",
            json={
                "model": "diffusion-jev",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
        assert r.status_code == 503
        assert "answers" not in r.json()


def test_question_ids_do_not_enter_prompt():
    req = EvaluationRequest.model_validate(
        {
            "model": "diffusion-jev",
            "state": "state",
            "questions": {"secret_id": {"type": "noul", "instructions": "question"}},
        }
    )
    assert "secret_id" not in json.dumps(messages(req.state, req.questions["secret_id"]))


def test_metrics_known_values():
    result = metrics([[1, 0], [0, 1]], [0, 1])
    assert result == {"accuracy": 1, "nll": 0, "brier": 0, "ece_10_bins": 0}
    uniform = metrics([[0.5, 0.5], [0.5, 0.5]], [0, 1])
    assert uniform["accuracy"] == 0.5 and uniform["brier"] == 0.5
    assert np.isclose(uniform["nll"], math.log(2))


def test_calibration_excludes_test_and_improves_dev(tmp_path):
    report = tmp_path / "report.json"
    out = tmp_path / "calibration.json"
    row = {
        "split": "test",
        "passes": 1,
        "logits": [0, 3],
        "probabilities": {"a": 0.1, "b": 0.9},
        "label": "a",
    }
    report.write_text(json.dumps({"model": "test", "records": [row]}))
    with pytest.raises(ValueError, match="development"):
        fit_calibration(report, out)
    row["split"] = "dev"
    report.write_text(json.dumps({"model": "test", "records": [row]}))
    fit_calibration(report, out)
    result = json.loads(out.read_text())
    assert result["temperature"] > 1 and result["nll_after"] < result["nll_before"]


def test_patch_idempotence_and_version_drift(tmp_path):
    scheduler = tmp_path / "srt/dllm/mixin/scheduler.py"
    scheduler.parent.mkdir(parents=True)
    (tmp_path / "srt/dllm/algorithm").mkdir()
    routing = tmp_path / "srt/layers/moe/topk.py"
    routing.parent.mkdir(parents=True)
    routing.write_text("        and fused_topk_deepseek is not None\n")
    scheduler.write_text(ANCHOR)
    patch(tmp_path)
    first = scheduler.read_text()
    patch(tmp_path)
    assert scheduler.read_text() == first and first.count(MARKER) == 1
    scheduler.write_text("incompatible")
    with pytest.raises(RuntimeError):
        patch(tmp_path)


@pytest.mark.asyncio
async def test_backend_requires_real_metadata_and_matching_passes():
    import httpx

    from diffusion_jev.backend import SGLangBackend

    class Tokenizer:
        def encode(self, label, **kwargs):
            return [ord(label)]

        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["return_dict"] is False
            return [10, 20, 30]

    backend = SGLangBackend("http://engine", Tokenizer())
    await backend.client.aclose()
    payload = {"meta_info": {"prompt_tokens": 3, "completion_tokens": 1}}

    def respond(req):
        assert json.loads(req.content)["sampling_params"]["max_new_tokens"] == 1
        return httpx.Response(200, json=payload)

    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    question = Noul(type="noul", instructions="?")
    with pytest.raises(BackendError, match="metadata"):
        await backend.score("x", question)
    payload["meta_info"].update(jev_candidate_logits=[list(range(26))], jev_passes=[2])
    with pytest.raises(BackendError, match="pass setting"):
        await backend.score("x", question)
    payload["meta_info"]["jev_passes"] = [1]
    result = await backend.score("x", question)
    assert result["logits"] == [0, 1]
    for invalid in (
        [],
        {"meta_info": []},
        {"meta_info": {"jev_candidate_logits": ["x" * 26]}},
        {"meta_info": {"jev_candidate_logits": [[float("inf")] * 26]}},
    ):
        payload = invalid
        with pytest.raises(BackendError):
            await backend.score("x", question)
    await backend.close()
