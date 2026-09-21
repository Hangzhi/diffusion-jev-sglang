"""Run yourself in a terminal to enter the key without echoing or shell history."""

import getpass
import os
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "web/.env.local"
key = getpass.getpass("AI Gateway API key (hidden): ").strip()
if not key or any(c in key for c in "\r\n\"' "):
    raise SystemExit("Empty or invalid key")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.fchmod(fd, 0o600)
with os.fdopen(fd, "w") as output:
    output.write("AI_GATEWAY_API_KEY=" + key + "\n")
print("Saved to ignored web/.env.local")
