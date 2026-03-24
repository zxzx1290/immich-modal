# immich-modal

Offload [Immich](https://immich.app/) machine learning inference to [Modal](https://modal.com/) serverless GPUs — CLIP smart search, face detection, and OCR.

## Why

Immich's ML workloads (CLIP embedding, face recognition, OCR) are GPU-hungry but only run in bursts. Running a dedicated GPU server 24/7 is wasteful. This project lets you run the Immich ML server on Modal's on-demand GPU infrastructure — you only pay when inference is actually happening.

## How it works

```
Immich (self-hosted)  ──HTTP──▶  Modal GPU container
                                  ├─ CLIP (smart search)
                                  ├─ Face detection
                                  └─ OCR
```

- Modal spins up a GPU container on demand when Immich sends an inference request
- The container runs the official Immich ML server (`immich_ml`) inside a custom Docker image
- Model weights are cached in a Modal Volume so they don't re-download on every cold start
- The container scales down automatically after `SCALEDOWN_WINDOW` seconds of idle time

## Requirements

- [Modal](https://modal.com/) account (free tier available)
- Immich instance (self-hosted)
- `modal` Python package

## Setup

### 1. Install and authenticate Modal

```bash
pip install modal
modal setup
```

### 2. Clone this repository

```bash
git clone https://github.com/your-username/immich-modal.git
cd immich-modal
```

### 3. Deploy

```bash
# Test mode (stops when terminal closes)
modal serve immich_ml_modal.py

# Permanent deployment (get a fixed URL)
modal deploy immich_ml_modal.py
```

After deploying, Modal will print a URL like:
```
https://your-username--immich-machine-learning-serve.modal.run
```

### 4. Configure Immich

In your Immich instance:

**Administration → System Settings → Machine Learning → URL**

Paste the Modal URL from the previous step.

> **Note:** The first request after a cold start may take a minute or two while the container boots and models load from the cache volume.

## Configuration

Edit the constants at the top of `immich_ml_modal.py`:

| Variable | Default | Description |
|---|---|---|
| `IMMICH_VERSION` | `"release"` | Immich ML image tag to use. Pin to a specific version (e.g. `"v1.132.3"`) for reproducibility. |
| `GPU_CONFIG` | `"T4"` | Modal GPU type. `T4` (16 GB) is cheapest; `L4` or `A10G` (24 GB) for heavier workloads. |
| `SCALEDOWN_WINDOW` | `300` | Seconds of idle time before the container scales down. |
| `ML_PORT` | `3003` | Port the Immich ML server listens on. No need to change. |

## Model cache

Model weights are stored in a Modal Volume named `immich-ml-model-cache`. This volume is created automatically on first deploy. Subsequent cold starts load models from the volume instead of re-downloading them from the internet.

## Disclaimer

This is an unofficial community project and is not affiliated with or endorsed by the Immich team or Modal.

## License

MIT
