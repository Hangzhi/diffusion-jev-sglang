"""Deploy the public demo on cheap CPU workers and one sleeping GPU.

Prepare data: python scripts/export_cloud_gallery.py
Download weights (CPU only): modal run deployment/modal_app.py::prepare_model
Deploy: modal deploy deployment/modal_app.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/diffusiongemma-26B-A4B-it"
MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
SGLANG_REVISION = "ef90bf677e4e6b280bf796f1c5f4ef0f05dda27f"
MODEL_PATH = "/models/diffusiongemma"
app = modal.App("diffusion-jev")
weights = modal.Volume.from_name("diffusion-jev-model", create_if_missing=True)
records = modal.Dict.from_name("diffusion-jev-demo-records", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.141.1",
        "httpx==0.28.1",
        "numpy==2.3.5",
        "pillow==11.3.0",
        "transformers==5.12.1",
        "pyyaml==6.0.3",
    )
    .env({"PYTHONPATH": "/app/src", "DIFFUSION_JEV_DATA_DIR": "/app/gallery"})
    .add_local_dir(ROOT / "src", "/app/src")
    .add_local_dir(ROOT / ".cache/cloud-gallery", "/app/gallery")
)

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "libnuma1", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install_from_requirements(ROOT / "deployment/gemma-requirements.txt")
    .env(
        {
            "SGLANG_BUILD_RUST_EXTS": "none",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0+diffusionjev",
            "PYTHONPATH": "/app/src:/opt/sglang/python",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .run_commands(
        "git init /opt/sglang",
        "git -C /opt/sglang remote add origin https://github.com/sgl-project/sglang.git",
        f"git -C /opt/sglang fetch --depth 1 origin {SGLANG_REVISION}",
        "git -C /opt/sglang checkout FETCH_HEAD",
        "python -m pip install --no-deps --no-build-isolation /opt/sglang/python",
    )
    .add_local_dir(
        ROOT / "src",
        "/app/src",
        copy=True,
        ignore=["**/static/**", "**/cloud.py", "**/__pycache__/**"],
    )
    .add_local_file(
        ROOT / "scripts/install_gemma_readout.py",
        "/app/scripts/install_gemma_readout.py",
        copy=True,
    )
    .run_commands(
        "mkdir -p /app/src/diffusion_jev/static",
        "python /app/scripts/install_gemma_readout.py /opt/sglang/python/sglang",
    )
    .add_local_file(
        ROOT / "scripts/launch_diffusiongemma.py", "/app/scripts/launch_diffusiongemma.py"
    )
)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==1.8.0"),
    volumes={"/models": weights},
    cpu=2,
    memory=4096,
    timeout=1800,
    max_containers=1,
    retries=0,
)
def prepare_model():
    """An explicit CPU-only download. Never download 50 GB on a metered GPU."""
    import json

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir=MODEL_PATH, max_workers=4)
    index = json.loads(Path(MODEL_PATH, "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        assert Path(MODEL_PATH, name).stat().st_size > 0, name
    Path(MODEL_PATH, "READY").write_text(MODEL_REVISION)
    weights.commit()
    return {"model": MODEL, "revision": MODEL_REVISION, "ready": True}


@app.cls(
    image=gpu_image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    volumes={"/models": weights},
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=60,
    startup_timeout=600,
    timeout=120,
    retries=0,
)
class Gemma:
    @modal.enter()
    def start(self):
        import subprocess
        import sys
        import time

        import httpx

        if Path(MODEL_PATH, "READY").read_text() != MODEL_REVISION:
            raise RuntimeError("Run prepare_model before starting the GPU.")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "/app/scripts/launch_diffusiongemma.py",
                "--model-path",
                MODEL_PATH,
                "--sglang-source",
                "/opt/sglang",
                "--engine-python",
                sys.executable,
            ],
            start_new_session=True,
        )
        deadline = time.monotonic() + 540
        with httpx.Client(timeout=3) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("The model process exited during startup.")
                try:
                    if client.get("http://127.0.0.1:8000/health").json().get("ready"):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(2)
        self.stop()
        raise TimeoutError("Model startup exceeded nine minutes.")

    @modal.method()
    def predict(self, payload):
        import httpx

        with httpx.Client(timeout=100) as client:
            response = client.post("http://127.0.0.1:8000/v1/systemone", json=payload)
            response.raise_for_status()
            return response.json()

    @modal.exit()
    def stop(self):
        import signal
        import subprocess

        if getattr(self, "process", None) and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)


class Store:
    async def put_if_absent(self, key, value):
        return await records.put.aio(key, value, skip_if_exists=True)

    async def put(self, key, value):
        await records.put.aio(key, value)

    async def get(self, key):
        return await records.get.aio(key, None)

    async def delete(self, key):
        await records.pop.aio(key, None)


class Worker:
    async def submit(self, payload):
        call = await Gemma().predict.spawn.aio(payload)
        return call.object_id

    async def result(self, call_id):
        import logging

        try:
            call = modal.FunctionCall.from_id(call_id)
            result = await call.get.aio(timeout=0)
            return {"status": "completed", "result": result}
        except TimeoutError:
            return {"status": "pending"}
        except Exception:
            logging.getLogger(__name__).exception("GPU prediction failed")
            return {
                "status": "failed",
                "detail": "The model could not finish this prediction. Please try again later.",
            }


@app.function(
    image=cpu_image,
    cpu=0.25,
    memory=512,
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
    timeout=30,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from diffusion_jev.cloud import Jobs, create_cloud_app

    return create_cloud_app(Jobs(Store(), Worker(), monthly=1000, daily=100))
