"""Exercise the actual patched SGLang admission method without importing its GPU runtime.

SGLANG_SOURCE=/path/to/pinned/sglang pytest tests/test_gemma_admission_integration.py
"""

import ast
import os
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffusion_jev.sglang_extension.gemma_prefill import jev_context_admission_tokens


def test_patched_scheduler_advertises_exact_context_length():
    directory = os.environ.get("SGLANG_SOURCE")
    if not directory:
        pytest.skip("Set SGLANG_SOURCE to validate the installed upstream scheduler")
    source = Path(directory) / "python/sglang/srt/managers/schedule_policy.py"
    tree = ast.parse(source.read_text())
    adder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PrefillAdder"
    )
    method = next(
        node
        for node in adder.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_prefill_admission"
    )
    scope = {
        "Req": object,
        "Optional": __import__("typing").Optional,
        "_PrefillAdmission": namedtuple(
            "Admission", "prefix_len extend_len max_new_tokens is_chunked"
        ),
        "CLIP_MAX_NEW_TOKENS": 4096,
        "AddReqResult": type("AddReqResult", (), {"OTHER": "other", "NO_TOKEN": "no_token"}),
        "jev_context_admission_tokens": jev_context_admission_tokens,
    }
    module = ast.Module(body=[method], type_ignores=[])
    # This is the local, hash-checked runtime source installed by our patcher.
    exec(compile(module, str(source), "exec"), scope)  # noqa: S102
    fake = SimpleNamespace(
        rem_total_tokens=16384,
        ceil_paged_tokens=lambda n: n,
        exact_chunk_fill=False,
        rem_chunk_tokens=None,
        is_hybrid_swa=False,
        can_run_list=[],
        rem_input_tokens=16384,
        rem_dllm_tokens=16384,
        dllm_block_size=256,
        page_size=1,
        dllm_config=SimpleNamespace(requires_separate_context_encoding=True),
        _check_prefill_tile_budget=lambda n: None,
        _get_dllm_remain_tokens=lambda req: 8192,
    )
    for size in (1, 90, 255, 256, 600):
        req = SimpleNamespace(
            prefix_indices=[],
            full_untruncated_fill_ids=list(range(size)),
            dllm_block_offset=size,
            is_dllm_prefill=lambda: True,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        planned = scope["_select_prefill_admission"](
            fake,
            req,
            host_hit_length=0,
            swa_host_hit_length=0,
            total_tokens=size + 1,
            truncation_align_size=None,
        )
        assert planned.extend_len == size
