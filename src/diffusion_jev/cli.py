import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import typer
import uvicorn

from .backend import MODEL, REVISION, SGLangBackend, candidate_ids

app = typer.Typer(no_args_is_help=True)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    engine_port: int = 30000,
    engine_url: str | None = None,
    engine_python: str = sys.executable,
    passes: int = typer.Option(1, min=1, max=4),
    temperature: float = typer.Option(1.0, min=0.001),
    calibration: Path | None = None,
    startup_timeout: int = 1800,
    wait_for_engine: bool = True,
):
    """Start the real model and local playground; Ctrl+C stops both."""
    from transformers import AutoTokenizer

    from .api import create_app

    if not wait_for_engine and not engine_url:
        raise typer.BadParameter("--no-wait-for-engine requires an external --engine-url")
    if passes not in (1, 2, 4):
        raise typer.BadParameter("passes must be 1, 2, or 4")
    if calibration:
        settings = json.loads(calibration.read_text())
        if settings["model"] != MODEL or settings["passes"] != passes:
            raise typer.BadParameter("Calibration model/passes do not match this run")
        temperature = float(settings["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise typer.BadParameter("temperature must be positive and finite")
    if not engine_url and engine_python == sys.executable:
        import importlib.metadata

        try:
            importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError:
            project = Path(__file__).resolve().parents[2]
            if not (project / "uv.lock").exists():
                raise typer.BadParameter(
                    "Install engine dependencies with uv pip install 'diffusion-jev-sglang[engine]'"
                )
            typer.echo("Installing the pinned SGLang engine on first run…")
            subprocess.run(
                ["uv", "sync", "--locked", "--extra", "engine", "--project", str(project)],
                check=True,
            )
    typer.echo("Loading pinned LLaDA tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    process = None
    config_path = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    url = engine_url or f"http://127.0.0.1:{engine_port}"
    try:
        if not engine_url:
            # Reject an occupied port instead of accidentally attaching to another engine.
            import socket

            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", engine_port))
                except OSError as error:
                    raise typer.BadParameter(
                        "Engine port is occupied; use --engine-url to attach explicitly"
                    ) from error
            setup = Path(__file__).with_name("engine_setup.py")
            subprocess.run([engine_python, str(setup)], check=True)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config:
                json.dump(
                    {"candidate_token_ids": candidate_ids(tokenizer), "passes": passes}, config
                )
                config_path = config.name
            command = [
                engine_python,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL,
                "--revision",
                REVISION,
                "--dllm-algorithm",
                "JevScoring",
                "--dllm-algorithm-config",
                config_path,
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "flashinfer",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "8",
                "--context-length",
                "8192",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
                "--disable-cuda-graph",
                "--host",
                "127.0.0.1",
                "--port",
                str(engine_port),
            ]
            typer.echo("Starting SGLang (first launch downloads ~32 GB of model weights)…")
            engine_env = os.environ.copy()
            engine_env["PATH"] = (
                str(Path(engine_python).parent) + os.pathsep + engine_env.get("PATH", "")
            )
            process = subprocess.Popen(command, start_new_session=True, env=engine_env)
        deadline = time.monotonic() + startup_timeout
        with httpx.Client(timeout=2) as client:
            while wait_for_engine:
                if process and process.poll() is not None:
                    raise RuntimeError("SGLang exited during startup; inspect the engine log above")
                try:
                    if client.get(url + "/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("SGLang readiness timed out")
                time.sleep(1)
        backend = SGLangBackend(url, tokenizer, expected_passes=passes)
        typer.echo(f"Playground: http://{host}:{port}   API docs: http://{host}:{port}/docs")
        uvicorn.run(create_app(backend, temperature), host=host, port=port)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        stop_process(process)
        if config_path:
            Path(config_path).unlink(missing_ok=True)


@app.command()
def benchmark(
    data: Path,
    output: Path = Path("reports/benchmark.json"),
    url: str = "http://127.0.0.1:8000",
    warmup: int = 2,
):
    """Evaluate labeled JSONL and record accuracy/calibration/end-to-end latency."""
    from .evaluation import run_benchmark

    run_benchmark(data, output, url, warmup)
    typer.echo(str(output))


@app.command()
def calibrate(report: Path, output: Path = Path("calibration.json")):
    """Fit temperature on a development-split benchmark report only."""
    from .evaluation import fit_calibration

    fit_calibration(report, output)
    typer.echo(str(output))


@app.command()
def doctor():
    """Check GPU, engine installation, and pinned model configuration."""
    import importlib.metadata

    checks = {"model": MODEL, "revision": REVISION, "python": sys.version.split()[0]}
    try:
        checks["sglang"] = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        checks["sglang"] = "missing: uv sync --extra engine"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    checks["gpu"] = result.stdout.strip() or result.stderr.strip()
    typer.echo(json.dumps(checks, indent=2))


@app.command("serve-gemma")
def serve_gemma(
    model_path: Path = Path("/workspace/models/diffusiongemma-26B-A4B-it"),
    engine_url: str = "http://127.0.0.1:30000",
    host: str = "127.0.0.1",
    port: int = 8000,
    denoising_steps: int = typer.Option(48, min=1, max=256),
):
    """Serve the Jev API and image playground using a running DiffusionGemma engine."""
    from transformers import PreTrainedTokenizerFast

    from .api import create_app
    from .gemma_backend import DiffusionGemmaBackend

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
    backend = DiffusionGemmaBackend(engine_url, tokenizer, expected_passes=denoising_steps)
    uvicorn.run(create_app(backend), host=host, port=port)
