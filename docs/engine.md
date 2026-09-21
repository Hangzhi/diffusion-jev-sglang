# Engine design and reproducibility

Pinned engine: SGLang **0.5.12.post1**, upstream tag commit `5a15cde858ea09b77116212a39356f2fc51b8584`.
Pinned checkpoint: `inclusionAI/LLaDA2.1-mini` revision `20e64e2ad21644d0e5248586ed9c942cdd45de0f`.

The [official model](https://huggingface.co/inclusionAI/LLaDA2.1-mini) is a roughly 16B total-parameter MoE diffusion model. The [SGLang recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/InclusionAI/LLaDA-2.1.mdx) provides the starting deployment configuration. This project replaces the normal denoising algorithm with `JevScoring` for direct decision reads.

1. Render the checkpoint's chat template with a state, question, and A–Z labeled options. Validate each candidate is a distinct single token. The pinned tokenizer assigns IDs 32–57.
2. Submit token IDs to native SGLang `/generate`, requesting one output token.
3. SGLang prepares its normal block-diffusion prefix and trailing mask block.
4. `JevScoring` forwards that block and reads the 26 candidate logits at its first masked position. With two/four passes it fills the other suffix positions greedily but **keeps the answer masked until the last read**; it never measures certainty after revealing the answer itself.
5. The scheduler attaches those floats to `meta_info.jev_candidate_logits` using SGLang's existing customized-info transport. No full vocabulary tensor crosses the HTTP boundary.
6. The API restricts to the question's actual candidate count, applies softmax(logits / temperature), and computes the typed answer.

The API emits raw logits in `meta.candidate_logits` for auditable calibration. This does not expose model reasoning.

Only BF16 and single-GPU serving are targeted. Radix reuse, overlap scheduling, and CUDA graphs are disabled in the initial correctness-oriented configuration. A batch contains separate question branches; repeated state tokens count separately in usage. Completion usage comes from SGLang and does not measure FLOPs or the number of denoising forwards. One answer token can require a block forward.

## Local SGLang branch

The supplied scheduler patch targets the pinned tag. The algorithm is an additional file maintained in this repository.

```bash
git -C /path/to/sglang worktree add -b feat/llada2-decision-scoring \
  /path/to/sglang-jev v0.5.12.post1
uv run python src/diffusion_jev/engine_setup.py /path/to/sglang-jev/python/sglang
# Install that checkout in an isolated engine environment, then use --engine-python.
```

`patches/sglang-0.5.12.post1-metadata.patch` contains the scheduler diff and an A100 routing fix: the FlashInfer fused DeepSeek top-k kernel is selected only on SM90+, so SM80 falls through to SGLang's existing compatible `moe_fused_gate` implementation. A live A100 warmup exposed this upstream dispatch issue. `src/diffusion_jev/sglang_extension/jev_scoring.py` contains the algorithm. The installer is idempotent and rejects an unrecognized anchor or unsupported installed engine version. Reinstall SGLang to remove the installed extension. Existing unrelated checkouts are not modified by the local-branch workflow.

A100 startup may compile FlashInfer kernels. Slow shared/network storage can dominate initial Python imports and weight loading; latency reports exclude startup.
