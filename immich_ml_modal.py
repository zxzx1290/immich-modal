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

GPU_CONFIG = "T4"            # T4=16GB cheapest; L4=24GB faster; A10G=24GB high load
# GPU_CONFIG = "L4"
# GPU_CONFIG = "A10G"

SCALEDOWN_WINDOW = 120       # seconds idle before sleep
MODEL_TTL = 90               # seconds to keep model in memory after last use
CONCURRENT_INPUTS = 4        # max concurrent requests per container
FUNCTION_TIMEOUT = 120       # seconds before Modal kills the function
STARTUP_TIMEOUT = 120        # seconds allowed for web server to start (≤ FUNCTION_TIMEOUT)

# ── Image ─────────────────────────────────────────────────────────────────────

dockerfile_path = Path(__file__).parent / "Dockerfile.immich-modal"

image = modal.Image.from_dockerfile(dockerfile_path).pip_install(
    "httpx",
    "fastapi",
    "uvicorn",
)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App(
    name="immich-machine-learning",
    image=image,
    secrets=[modal.Secret.from_name("immich-proxy-key")],
)

# ── ASGI App factory (auth + reverse proxy) ───────────────────────────────────
# fastapi is only available inside the Modal container image, NOT on the local
# machine running `modal deploy`.  Importing it at module-level would cause a
# ModuleNotFoundError during the deploy step.  Wrapping the imports inside a
# factory function defers them until the code actually runs inside the container.

def _make_web_app():
    from fastapi import FastAPI, Request, Response  # container-only import
    import httpx

    web_app = FastAPI()

    @web_app.middleware("http")
    async def verify_key(request: Request, call_next):
        key = request.headers.get("X-Modal-Proxy-Key", "")
        if key != os.environ.get("MODAL_PROXY_KEY", ""):
            return Response(
                content='{"error":"forbidden"}',
                status_code=403,
                media_type="application/json",
            )
        return await call_next(request)

    @web_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        body = await request.body()
        skip = {"host", "x-modal-proxy-key", "transfer-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=f"http://127.0.0.1:3003/{path}",
                headers=headers,
                content=body,
                timeout=FUNCTION_TIMEOUT,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    return web_app

# ── Web Endpoint ──────────────────────────────────────────────────────────────


@app.function(
    gpu=GPU_CONFIG,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=FUNCTION_TIMEOUT,
    volumes={
        "/cache": modal.Volume.from_name(
            "immich-ml-model-cache",
            create_if_missing=True,
        )
    },
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS)
@modal.asgi_app()
def serve():
    """
    Start the Immich ML server as a subprocess, fronted by a FastAPI auth proxy.

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
        "MACHINE_LEARNING_MODEL_TTL": str(MODEL_TTL),
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
                "http://127.0.0.1:3003/ping", timeout=2
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

    print("Waiting for Immich ML server on port 3003...")
    while not ml_server_ready():
        time.sleep(2.0)
    print("Immich ML server is ready.")

    return _make_web_app()


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    print(
        "\nCommands:\n"
        "  Test:   modal serve immich_ml_modal.py\n"
        "  Deploy: modal deploy immich_ml_modal.py\n"
    )
