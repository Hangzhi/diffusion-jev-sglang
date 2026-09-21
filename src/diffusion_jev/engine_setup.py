"""Install a small, version-checked patch into the selected SGLang environment."""

import importlib.metadata
import importlib.util
import os
import tempfile
from pathlib import Path

VERSION = "0.5.12.post1"
MARKER = "# diffusion-jev: transport real candidate logits"
ANCHOR = "                if new_tokens == 0:\n                    continue\n"
INSERT = """                # diffusion-jev: transport real candidate logits
                info = result.logits_output.customized_info
                if info is not None:
                    if req.customized_info is None:
                        req.customized_info = {}
                    for key, values in info.items():
                        req.customized_info.setdefault(key, []).append(values[idx])
"""


def atomic_write(path, content):
    # Replacing the inode avoids changing hardlinked uv-cache or other environments.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        temp.write(content)
        temporary = temp.name
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def patch(root: Path | None = None):
    if root is None:
        if importlib.metadata.version("sglang") != VERSION:
            raise RuntimeError(f"Expected SGLang {VERSION}; use uv sync --extra engine")
        spec = importlib.util.find_spec("sglang")
        root = Path(next(iter(spec.submodule_search_locations)))
    target = root / "srt/dllm/mixin/scheduler.py"
    source = target.read_text()
    old_anchor = "                req = batch.reqs[idx]\n"
    source = source.replace(old_anchor + INSERT, old_anchor)
    if MARKER not in source:
        if source.count(ANCHOR) != 1:
            raise RuntimeError("Unsupported SGLang scheduler: patch anchor mismatch")
        source = source.replace(ANCHOR, ANCHOR + INSERT)
        atomic_write(target, source)
    routing = root / "srt/layers/moe/topk.py"
    routing_source = routing.read_text()
    routing_anchor = "        and fused_topk_deepseek is not None\n"
    guard = "        and torch.cuda.get_device_capability(gating_output.device)[0] >= 9  # diffusion-jev: SM90+ only\n"
    if "# diffusion-jev: SM90+ only" not in routing_source:
        if routing_source.count(routing_anchor) != 1:
            raise RuntimeError("Unsupported SGLang MoE routing: patch anchor mismatch")
        atomic_write(routing, routing_source.replace(routing_anchor, routing_anchor + guard))
    extension = Path(__file__).parent / "sglang_extension/jev_scoring.py"
    atomic_write(root / "srt/dllm/algorithm/jev_scoring.py", extension.read_text())


if __name__ == "__main__":
    import sys

    patch(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
