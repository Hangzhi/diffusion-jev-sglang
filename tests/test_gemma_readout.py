"""The readout must preserve an early-finished row while its batch continues."""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_readout_uses_last_active_step_without_changing_sampler_output():
    class Native:
        def __init__(self, config):
            self.block_size = 6
            self.vocab_size = 30
            self.max_denoising_steps = 2

        def step(self, batch, logits, states):
            for i, state in enumerate(states):
                if not state["finished"]:
                    state["step"] -= 1
                    state["finished"] = i == 0 or state["step"] == 0
            return [s["finished"] for s in states]

        def run(self, runner, batch, algo_states=None):
            states = [{"finished": False, "step": 2} for _ in range(2)]
            first = torch.zeros(2, 6, 30)
            for position, token in enumerate([26, 27, 28, 29]):
                first[:, position, token] = 20
            first[0, 4, 3] = 7
            first[1, 4, 4] = 8
            self.step(batch, first, states)
            second = first.clone()
            second[0, 4, 9] = 99  # This row already finished: discard this later readout.
            second[1, 4, 5] = 10
            self.step(batch, second, states)
            return SimpleNamespace(customized_info=None), [[3], [5]], None, None, False

    scope = {"Gemma4Renoise": Native}
    source = Path(__file__).parents[1] / "src/diffusion_jev/sglang_extension/gemma_readout.py"
    # Execute the local extension against a tiny sampler to verify batched readout.
    exec(compile(source.read_text(), str(source), "exec"), scope)  # noqa: S102
    config = SimpleNamespace(
        algorithm_config={
            "jev_candidate_token_ids": list(range(26)),
            "jev_answer_prefix_token_ids": [26, 27, 28, 29],
        }
    )
    native = Native(config).run(None, SimpleNamespace(batch_size=2))
    observed = scope["Algorithm"](config).run(None, SimpleNamespace(batch_size=2))
    assert observed[1:] == native[1:]
    info = observed[0].customized_info
    assert info["jev_denoising_steps"] == [1, 2]
    assert info["jev_unrestricted_token"] == [3, 5]
    assert info["jev_answer_prefix_tokens"] == [[26, 27, 28, 29]] * 2
    assert info["jev_candidate_logits"][0][3] == 7
    assert info["jev_candidate_logits"][0][9] == 0
    assert info["jev_candidate_logits"][1][5] == 10
