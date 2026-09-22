"""Prepare Vercel's static Build Output; only job requests reach Modal.

Run after npm run build and export_cloud_gallery.py. Then vercel deploy --prebuilt.
"""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def build(backend, output):
    parsed = urlparse(backend)
    if parsed.scheme != "https" or not parsed.hostname.endswith(".modal.run"):
        raise ValueError("Use the deployed HTTPS Modal web endpoint")
    if output.exists():
        shutil.rmtree(output)
    public = output / "static"
    shutil.copytree(ROOT / "src/diffusion_jev/static", public)
    shutil.copytree(ROOT / ".cache/cloud-gallery", public / "gallery")
    shutil.copy2(ROOT / "src/diffusion_jev/data/presets.json", public / "examples.json")
    (public / "cloud-health.json").write_text(
        json.dumps(
            {
                "ready": True,
                "model": "google/diffusiongemma-26B-A4B-it",
                "display_name": "DiffusionGemma 26B A4B",
                "supports_images": True,
                "execution": "queued",
            "gpu_policy": "starts_on_request",
            "api_docs": backend.rstrip("/") + "/docs",
            }
        )
    )
    (output / "config.json").write_text(
        json.dumps(
            {
                "version": 3,
                "routes": [
                    {"src": "/health", "dest": "/cloud-health.json"},
                    {"src": "/api/examples", "dest": "/examples.json"},
                    {"src": "/api/jobs(/.*)?", "dest": backend.rstrip("/") + "/api/jobs$1"},
                    {"handle": "filesystem"},
                    {"src": "/.*", "dest": "/index.html"},
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Vercel output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".vercel/output")
    args = parser.parse_args()
    build(args.backend, args.output)
