"""Context admission for the pinned PR's non-ignore-EOS scheduling path.

Context encoding uses the actual available prompt tokens. Denoising requires
a complete canvas. A short prompt must never be advertised as 256 tokens.
"""


def jev_context_admission_tokens(adder, req, prefix_len):
    available = adder._get_dllm_remain_tokens(req)
    if req.is_dllm_prefill():
        return min(available, req.dllm_block_offset - prefix_len)
    remaining = len(req.full_untruncated_fill_ids) - prefix_len
    return adder.dllm_block_size if min(available, remaining) >= adder.dllm_block_size else 0
