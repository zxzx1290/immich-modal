"""
Immich Remote Machine Learning on Modal
========================================
Offloads Immich ML inference (CLIP Smart Search, Face Detection, OCR)
to a Modal cloud GPU.

Requirements:
  - This .py and Dockerfile.immich-modal must be in the same directory.

Usage:
  1. pip install modal
  2. modal setup
  3. modal serve immich_ml_modal.py      <- test, stops when terminal closes
  4. modal deploy immich_ml_modal.py     <- permanent deploy, get a fixed URL
  5. Add the URL in Immich:
     Administration -> System Settings -> Machine Learning -> Add URL
"""

import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

# ── Config ────────────────────────────────────────────────────────────────────

IMMICH_VERSION = "release"   # or pin e.g. "v1.132.3"
ML_PORT = 3003

GPU_CONFIG = "T4"            # T4=16GB cheapest; L4=24GB faster; A10G=24GB high load
# GPU_CONFIG = "L4"
# GPU_CONFIG = "A10G"

SCALEDOWN_WINDOW = 120       # seconds idle before sleep

# ── Image ─────────────────────────────────────────────────────────────────────

dockerfile_path = Path(__file__).parent / "Dockerfile.immich-modal"

image = modal.Image.from_dockerfile(dockerfile_path)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App(
    name="immich-machine-learning",
    image=image,
)

# ── Web Endpoint ──────────────────────────────────────────────────────────────

@app.function(
    gpu=GPU_CONFIG,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=120,
    volumes={
        "/cache": modal.Volume.from_name(
            "immich-ml-model-cache",
            create_if_missing=True,
        )
    },
)
@modal.web_server(port=ML_PORT, startup_timeout=120)  # match function timeout
def serve():
    """
    Start the Immich ML server as a subprocess.

    Key design: Modal's runtime runner uses the system Python (/usr/local/bin/python).
    We must NOT override PATH globally (done in Dockerfile) because that would make
    Modal's runner pick up the venv Python which lacks Modal's own deps (grpclib etc.).
    Instead, we pass the venv PATH explicitly only to the Immich subprocess.
    """
    # Build env for the Immich subprocess only - venv Python takes priority here
    immich_env = {
        **os.environ,
        "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "/usr/src",
        "VIRTUAL_ENV": "/opt/venv",
        "MACHINE_LEARNING_WORKERS": "1",
        "MACHINE_LEARNING_CACHE_FOLDER": "/cache",
        "TRANSFORMERS_CACHE": "/cache",
        "MACHINE_LEARNING_MODEL_TTL": "90",
        "DEVICE": "cuda",
        # mimalloc: find the actual .so path for this arch
        "LD_PRELOAD": (
            "/usr/lib/x86_64-linux-gnu/libmimalloc.so.2"
            if os.path.exists("/usr/lib/x86_64-linux-gnu/libmimalloc.so.2")
            else ""
        ),
    }

    # Sanity check: confirm required files exist before launching subprocess
    venv_python = shutil.which("python", path="/opt/venv/bin")
    immich_ml_dir = Path("/usr/src/immich_ml")
    immich_ml_main = immich_ml_dir / "__main__.py"
    print(f"[sanity] venv python    : {venv_python}")
    print(f"[sanity] immich_ml dir  : {immich_ml_dir.exists()} ({immich_ml_dir})")
    print(f"[sanity] immich_ml main : {immich_ml_main.exists()} ({immich_ml_main})")

    process = subprocess.Popen(
        ["/opt/venv/bin/python", "-m", "immich_ml"],
        env=immich_env,
        cwd="/usr/src",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout so nothing is lost
    )

    # Background thread: relay all subprocess output to Modal logs in real time
    def _relay_output():
        for line in process.stdout:
            print("[immich_ml]", line.decode(errors="replace"), end="", flush=True)

    threading.Thread(target=_relay_output, daemon=True).start()

    def ml_server_ready() -> bool:
        # Use HTTP /ping instead of bare TCP connect — port open ≠ server ready
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{ML_PORT}/ping", timeout=2
            ) as resp:
                return resp.status == 200
        except Exception:
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"Immich ML server exited unexpectedly (code {retcode}). "
                    "Check [immich_ml] lines above for the cause."
                )
            return False

    print(f"Waiting for Immich ML server on port {ML_PORT}...")
    while not ml_server_ready():
        time.sleep(2.0)

    print("Immich ML server is ready.")
    # serve() must return here so Modal's web_server proxy can start forwarding
    # traffic. The subprocess keeps running in the background; the output relay
    # thread (daemon=True) will print any crash output to Modal logs.


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print(
        "\nCommands:\n"
        "  Test:   modal serve immich_ml_modal.py\n"
        "  Deploy: modal deploy immich_ml_modal.py\n"
    )
