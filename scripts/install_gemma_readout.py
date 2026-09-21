"""Add readout metadata to an isolated checkout of SGLang PR 34061.

Use only with ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f. No sampler math is changed.
"""

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# diffusion-jev: Gemma denoiser readout"
SCHEDULER_INSERT = """
                # diffusion-jev: Gemma denoiser readout
                info = getattr(result.logits_output, "customized_info", None)
                resolved = not fdfo_mode or result.accept_length_per_req_cpu[idx] > 0
                if info is not None and resolved:
                    if req.customized_info is None:
                        req.customized_info = {}
                    for key, values in info.items():
                        req.customized_info.setdefault(key, []).append(values[idx])
"""
PREFILL_INSERT = """            # diffusion-jev: match actual context length before allocating attention metadata
            if self.dllm_config.requires_separate_context_encoding:
                extend_len = jev_context_admission_tokens(self, req, prefix_len)
"""


def install(package):
    sampler = package / "srt/dllm/algorithm/gemma4_renoise.py"
    scheduler = package / "srt/dllm/mixin/scheduler.py"
    policy = package / "srt/managers/schedule_policy.py"
    sampler_base = sampler.read_text().split("\n\n" + MARKER)[0]
    scheduler_base = scheduler.read_text().replace(SCHEDULER_INSERT, "")
    policy_base = policy.read_text().split("\n\n" + MARKER)[0].replace(PREFILL_INSERT, "")
    expected = (
        (sampler_base, "79e24d96c4db97ce29ddd45c9ee9c7e339d4aa1d399204b1d3c50488690ef092"),
        (scheduler_base, "3f2ce702f4be7922ffe8e1c726225aac2c3549794ffd6b0314d19e72475de7b1"),
        (policy_base, "72a94ec8520bb35e718e4ceda4a0348f4661d54ba3148ef97a924915bbf10f5b"),
    )
    if any(hashlib.sha256(source.encode()).hexdigest() != digest for source, digest in expected):
        raise ValueError("Source differs from the pinned SGLang PR revision; refusing to patch")
    extension = ROOT / "src/diffusion_jev/sglang_extension/gemma_readout.py"
    sampler.write_text(sampler_base + "\n\n" + MARKER + "\n" + extension.read_text())
    anchor = "                req = batch.reqs[idx]\n"
    scheduler.write_text(scheduler_base.replace(anchor, anchor + SCHEDULER_INSERT))
    anchor = """                * self.page_size
            )
            if extend_len <= 0:
"""
    if policy_base.count(anchor) != 1:
        raise ValueError("Unsupported prefill admission code")
    corrected = policy_base.replace(
        anchor,
        anchor.replace(
            "            if extend_len <= 0:\n",
            PREFILL_INSERT + "            if extend_len <= 0:\n",
        ),
    )
    helper = ROOT / "src/diffusion_jev/sglang_extension/gemma_prefill.py"
    policy.write_text(corrected + "\n\n" + MARKER + "\n" + helper.read_text())
    print("Installed denoiser readout in", package)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sglang_package", type=Path)
    install(parser.parse_args().sglang_package)
