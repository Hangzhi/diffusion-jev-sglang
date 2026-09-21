"""Supervise the pinned DiffusionGemma SGLang engine and local Jev web app.

The selected engine environment must contain the dependencies of SGLang PR 34061.
Use docs/diffusiongemma.md to prepare the source and checkpoint first.
"""

import argparse
import json
import os
import signal
import socket
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from install_gemma_readout import install

from diffusion_jev.gemma_backend import ANSWER_PREFIX

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main(args):
    from transformers import PreTrainedTokenizerFast

    for port in (args.port, args.engine_port):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    install(args.sglang_source / "python/sglang")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.model_path))
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in string.ascii_uppercase]
    if not all(len(item) == 1 for item in ids):
        raise ValueError("Each A–Z candidate must be a single token")
    engine = api = None
    configuration = None

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "jev_candidate_token_ids": [item[0] for item in ids],
                    "jev_answer_prefix_token_ids": tokenizer.encode(
                        ANSWER_PREFIX, add_special_tokens=False
                    ),
                    "max_denoising_steps": args.denoising_steps,
                    "seed": 42,
                },
                f,
            )
            configuration = f.name
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.sglang_source / "python")
        env["PATH"] = str(args.engine_python.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        # Poll process readiness without inserting synthetic one-token GPU work.
        # End-to-end health is verified separately with real classification requests.
        env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
        engine = subprocess.Popen(
            [
                str(args.engine_python),
                "-m",
                "sglang.launch_server",
                "--model-path",
                str(args.model_path),
                "--served-model-name",
                "google/diffusiongemma-26B-A4B-it",
                "--dllm-algorithm",
                "Gemma4Renoise",
                "--dllm-algorithm-config",
                configuration,
                "--no-dllm-fdfo",
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.88",
                "--max-running-requests",
                "4",
                "--context-length",
                "8192",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                *(["--skip-server-warmup"] if args.skip_server_warmup else []),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.engine_port),
            ],
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
        api = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from diffusion_jev.cli import app; app()",
                "serve-gemma",
                "--model-path",
                str(args.model_path),
                "--engine-url",
                f"http://127.0.0.1:{args.engine_port}",
                "--port",
                str(args.port),
                "--denoising-steps",
                str(args.denoising_steps),
            ],
            cwd=ROOT,
            start_new_session=True,
        )
        print(f"DiffusionGemma Jev: http://127.0.0.1:{args.port}", flush=True)
        while engine.poll() is None and api.poll() is None:
            time.sleep(1)
        raise RuntimeError(f"Service exited: engine={engine.poll()}, API={api.poll()}")
    except KeyboardInterrupt:
        pass
    finally:
        stop(api)
        stop(engine)
        if configuration:
            Path(configuration).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/workspace/models/diffusiongemma-26B-A4B-it")
    )
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--engine-python", type=Path, required=True)
    parser.add_argument("--denoising-steps", type=int, default=48)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-port", type=int, default=30000)
    parser.add_argument("--skip-server-warmup", action="store_true")
    main(parser.parse_args())
