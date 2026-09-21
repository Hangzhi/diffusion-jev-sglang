"""SGLang 0.5.12.post1 extension: read the first masked position, never generated confidence.

All other masked positions are filled between passes, while the answer stays masked.
The final read therefore never conditions on its own selected answer. Only candidate
logits are transported to the API. One request scores one question.
"""

import torch
from sglang.srt.dllm.algorithm.base import DllmAlgorithm


class JevScoring(DllmAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.candidate_ids = config.algorithm_config["candidate_token_ids"]
        self.passes = int(config.algorithm_config.get("passes", 1))
        if self.passes not in (1, 2, 4):
            raise ValueError("JevScoring supports 1, 2, or 4 passes")
        if len(self.candidate_ids) != 26 or len(set(self.candidate_ids)) != 26:
            raise ValueError("Expected 26 distinct single-token A-Z candidate IDs")

    def run(self, model_runner, forward_batch):
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # Prefill may contain many full blocks, not just one block per request.
        if not (forward_batch.input_ids == self.mask_id).any().item():
            return out.logits_output, [], out.can_run_graph
        batch_size = forward_batch.batch_size
        ids = forward_batch.input_ids.view(batch_size, self.block_size)
        mask = ids == self.mask_id
        starts = (~mask).sum(dim=1).tolist()
        rows = torch.arange(batch_size, device=ids.device)
        active_rows = mask.any(dim=1)
        slots = torch.tensor([min(s, self.block_size - 1) for s in starts], device=ids.device)
        for step in range(self.passes):
            if step:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits = out.logits_output.full_logits.view(batch_size, self.block_size, -1)
            candidate_logits = logits[rows, slots][:, self.candidate_ids].float().clone()
            filled = torch.where(mask, logits.argmax(-1), ids)
            if step < self.passes - 1:
                # Keep answer masked during every probability read.
                filled[rows[active_rows], slots[active_rows]] = self.mask_id
            ids.copy_(filled)
        out.logits_output.customized_info = {
            "jev_candidate_logits": candidate_logits.cpu().tolist(),
            "jev_passes": [self.passes] * batch_size,
        }
        next_ids = [ids[i, start:] for i, start in enumerate(starts)]
        return out.logits_output, next_ids, out.can_run_graph


Algorithm = JevScoring
