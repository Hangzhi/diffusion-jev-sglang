# Remote GPU, local Mac browser

For the current image-capable model, start the [DiffusionGemma launcher](diffusiongemma.md#launch-the-service) on the GPU machine. The earlier LLaDA backend uses:

```bash
cd /path/to/diffusion-jev-sglang
uv run diffusion-jev serve
```

On the Mac, forward a local port through the existing SSH host alias:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18000:127.0.0.1:8000 YOUR_GPU_SSH_ALIAS
```

Leave that terminal running, then open **http://localhost:18000/#doodle** in the Mac browser. The model stays on the A100, and browser/API traffic goes through SSH. Only port 8000 needs forwarding; the SGLang engine remains private on remote port 30000.

If 18000 is occupied, replace the first port with another free local port. If you connect with an explicit SSH port or identity file, use the same `-p` and `-i` flags as your normal connection. Start the tunnel on the Mac, not inside the remote SSH shell.

The CLI prints the playground URL after the engine is ready. A cold start can take several minutes, especially with dependencies and weights on shared network storage. Stopping the Mac tunnel does not stop inference; stop the remote CLI with Ctrl+C when finished.

To show the playground immediately while a separately managed engine starts, run:

```bash
uv run diffusion-jev serve --engine-url http://127.0.0.1:30000 --no-wait-for-engine
```

The page polls readiness and enables evaluation only when the engine responds successfully. Runtime details for this working session are in the ignored `.cache/run-info.json`; `.cache/api.log` and `.cache/engine.log` link to the running service logs.
