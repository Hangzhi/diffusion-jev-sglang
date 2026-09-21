"""Tensor-level extension tests; run in a torch environment (GPU is not required)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def algorithm(monkeypatch):
    name = "sglang.srt.dllm.algorithm.base"
    module = ModuleType(name)

    class Base:
        def __init__(self, config):
            self.block_size = config.block_size
            self.mask_id = config.mask_id

    module.DllmAlgorithm = Base
    monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).parents[1] / "src/diffusion_jev/sglang_extension/jev_scoring.py"
    spec = importlib.util.spec_from_file_location("test_jev_extension", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JevScoring


@pytest.mark.parametrize("passes", [1, 2, 4])
def test_answer_remains_masked_and_context_is_preserved(algorithm, passes):
    model = algorithm(
        SimpleNamespace(
            block_size=4,
            mask_id=99,
            algorithm_config={"candidate_token_ids": list(range(26)), "passes": passes},
        )
    )
    inputs = torch.tensor([71, 72, 99, 99, 81, 99, 99, 99])
    batch = SimpleNamespace(input_ids=inputs, batch_size=2)
    seen = []

    class Runner:
        def forward(self, batch, pp_proxy_tensors=None):
            seen.append(batch.input_ids.clone())
            logits = torch.zeros(8, 100)
            logits[:, 7] = 2
            return SimpleNamespace(
                logits_output=SimpleNamespace(full_logits=logits, customized_info=None),
                can_run_graph=False,
            )

    output, tokens, _ = model.run(Runner(), batch)
    assert len(seen) == passes
    assert all(x[2] == 99 and x[5] == 99 for x in seen)
    assert all(x[:2].tolist() == [71, 72] and x[4] == 81 for x in seen)
    assert [len(x) for x in tokens] == [2, 3]
    assert output.customized_info["jev_passes"] == [passes, passes]
    assert output.customized_info["jev_candidate_logits"][0][7] == 2
    assert all(x[3] == 7 for x in seen[1:])


@pytest.mark.parametrize("length", [4, 12, 128])
def test_prefill_only_is_not_a_decision(algorithm, length):
    model = algorithm(
        SimpleNamespace(
            block_size=4, mask_id=99, algorithm_config={"candidate_token_ids": list(range(26))}
        )
    )
    output = SimpleNamespace(full_logits=torch.zeros(length, 100), customized_info=None)
    runner = SimpleNamespace(
        forward=lambda *a, **kw: SimpleNamespace(logits_output=output, can_run_graph=False)
    )
    result, tokens, _ = model.run(
        runner, SimpleNamespace(input_ids=torch.ones(length, dtype=torch.long), batch_size=1)
    )
    assert tokens == [] and result.customized_info is None


def test_mixed_full_context_row_is_never_remasked(algorithm):
    model = algorithm(
        SimpleNamespace(
            block_size=4,
            mask_id=99,
            algorithm_config={"candidate_token_ids": list(range(26)), "passes": 4},
        )
    )
    inputs = torch.tensor([71, 72, 73, 74, 81, 99, 99, 99])
    seen = []

    class Runner:
        def forward(self, batch, pp_proxy_tensors=None):
            seen.append(batch.input_ids.clone())
            logits = torch.zeros(8, 100)
            logits[:, 7] = 2
            return SimpleNamespace(
                logits_output=SimpleNamespace(full_logits=logits, customized_info=None),
                can_run_graph=False,
            )

    _, tokens, _ = model.run(Runner(), SimpleNamespace(input_ids=inputs, batch_size=2))
    assert all(row[:4].tolist() == [71, 72, 73, 74] for row in seen)
    assert len(tokens[0]) == 0
