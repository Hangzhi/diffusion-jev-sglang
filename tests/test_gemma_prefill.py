from types import SimpleNamespace

from diffusion_jev.sglang_extension.gemma_prefill import jev_context_admission_tokens


def admission(prompt, *, prefix=0, budget=8192, prefill=True, canvas=256):
    adder = SimpleNamespace(_get_dllm_remain_tokens=lambda req: budget, dllm_block_size=canvas)
    req = SimpleNamespace(
        dllm_block_offset=prompt,
        full_untruncated_fill_ids=[0] * (prompt + canvas),
        is_dllm_prefill=lambda: prefill,
    )
    return jev_context_admission_tokens(adder, req, prefix)


def test_short_context_never_reads_past_actual_input():
    # The original normal admission path always claimed 256, even with one token.
    for length in (1, 15, 107, 255):
        assert admission(length) == length


def test_image_context_is_not_split_at_denoising_canvas_boundary():
    assert admission(600) == 600
    assert admission(600, prefix=256) == 344


def test_context_respects_available_kv_and_canvas_is_never_partial():
    assert admission(600, budget=300) == 300
    assert admission(600, prefix=600, prefill=False, budget=256) == 256
    assert admission(600, prefix=600, prefill=False, budget=255) == 0
