"""Appended to the pinned Gemma4Renoise sampler by install_gemma_readout.py.

Read genuine answer-position logits on the LAST active denoising step per row.
These are self-conditioned denoiser scores, not independent correctness estimates.
The sampler, generated tokens, stopping rule, and probability schedule are unchanged.
"""

_NativeGemma4Renoise = Gemma4Renoise  # noqa: F821


class Gemma4Renoise(_NativeGemma4Renoise):
    def __init__(self, config):
        super().__init__(config)
        self.jev_candidates = config.algorithm_config.get("jev_candidate_token_ids")
        if not self.jev_candidates or len(self.jev_candidates) != 26:
            raise ValueError("Jev readout requires 26 A–Z candidate token IDs")
        self.jev_prefix = config.algorithm_config.get("jev_answer_prefix_token_ids")
        if not self.jev_prefix or len(self.jev_prefix) >= self.block_size:
            raise ValueError("Jev readout requires the checkpoint's empty thought prefix")
        self.jev_states = None

    def step(self, forward_batch, full_logits, states):
        import torch

        logits = full_logits.view(forward_batch.batch_size, self.block_size, self.vocab_size)
        offset = len(self.jev_prefix)
        first = logits[:, offset, :].float()
        prefix = logits[:, :offset, :].argmax(-1)
        candidates = first[:, self.jev_candidates].clone()
        mass = (torch.logsumexp(candidates, -1) - torch.logsumexp(first, -1)).exp()
        for i, state in enumerate(states):
            if not state["finished"]:
                state["jev_logits"] = candidates[i]
                state["jev_candidate_mass"] = mass[i]
                state["jev_unrestricted_token"] = first[i].argmax()
                state["jev_answer_prefix_tokens"] = prefix[i].clone()
        done = super().step(forward_batch, full_logits, states)
        self.jev_states = states
        return done

    def run(self, model_runner, forward_batch, algo_states=None):
        self.jev_states = None
        result = super().run(model_runner, forward_batch, algo_states)
        if self.jev_states is not None:
            info = {
                "jev_candidate_logits": [],
                "jev_denoising_steps": [],
                "jev_candidate_mass": [],
                "jev_unrestricted_token": [],
                "jev_max_denoising_steps": [],
                "jev_answer_prefix_tokens": [],
            }
            for state in self.jev_states:
                info["jev_candidate_logits"].append(state["jev_logits"].cpu().tolist())
                info["jev_denoising_steps"].append(self.max_denoising_steps - state["step"])
                info["jev_candidate_mass"].append(state["jev_candidate_mass"].item())
                info["jev_unrestricted_token"].append(state["jev_unrestricted_token"].item())
                info["jev_max_denoising_steps"].append(self.max_denoising_steps)
                info["jev_answer_prefix_tokens"].append(
                    state["jev_answer_prefix_tokens"].cpu().tolist()
                )
            result[0].customized_info = info
        return result


Algorithm = Gemma4Renoise
