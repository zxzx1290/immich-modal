# immich-modal

Run [Immich](https://immich.app/) machine learning inference (CLIP, face detection, OCR) on [Modal](https://modal.com/) serverless GPUs.

## Architecture

```
Immich Server --> CF Worker (auth) --> Modal (FastAPI auth proxy --> Immich ML subprocess)
```

- Immich ML runs as a subprocess on a Modal GPU container
- A FastAPI ASGI layer verifies the `X-Modal-Proxy-Key` header before forwarding requests to the ML server
- Model weights are cached in a Modal Volume so they persist across cold starts
- The container scales down automatically after idle time

## Requirements

- [Modal](https://modal.com/) account (free tier available)
- `modal` Python package
- A Modal secret named `immich-proxy-key` containing `MODAL_PROXY_KEY` (and optionally `HF_TOKEN`)

## Setup

### 1. Install and authenticate Modal

```bash
pip install modal
modal setup
```

### 2. Create the shared secret

```bash
modal secret create immich-proxy-key \
  MODAL_PROXY_KEY=your-secret-key \
  HF_TOKEN=hf_xxxxxxxx
```

Every key in this secret is exported into the container environment and inherited
by the Immich ML subprocess.

`HF_TOKEN` is optional. Immich ML downloads model weights from Hugging Face, and
anonymous downloads share a per-IP rate limit - since Modal's egress IPs are
shared between tenants, a cold model cache can occasionally hit a 429. A
[read-only token][hf-tokens] avoids that and silences the `You are sending
unauthenticated requests to the HF Hub` warning in the logs.

[hf-tokens]: https://huggingface.co/settings/tokens

### 3. Deploy

```bash
# Test mode (stops when terminal closes)
modal serve immich_ml_modal.py

# Permanent deployment (get a fixed URL)
modal deploy immich_ml_modal.py
```

### 4. Configure Immich

In your Immich instance, go to **Administration > System Settings > Machine Learning** and set the URL to your Cloudflare Worker proxy URL (not the Modal URL directly).

> **Note:** The first request after a cold start may take a minute or two while the container boots and models load.

## Configuration

Edit the constants at the top of `immich_ml_modal.py`:

| Variable | Default | Description |
|---|---|---|
| `GPU_CONFIG` | `T4` | Modal GPU type (`T4`, `L4`, `A10G`) |
| `SCALEDOWN_WINDOW` | `120` | Seconds idle before container sleeps |
| `MODEL_TTL` | `90` | Seconds to keep ML model in memory |
| `CONCURRENT_INPUTS` | `4` | Max concurrent requests per container |
| `FUNCTION_TIMEOUT` | `120` | Seconds before Modal kills the function |

### Immich version

The Immich ML image is pinned in the `FROM` line of `Dockerfile.immich-modal`.
Immich recommends keeping it in sync with the server version - mismatches are
not rejected, but [may cause bugs and instability][docs]. To upgrade: bump the
server first, then update the tag here and run `modal deploy` again.

[docs]: https://docs.immich.app/guides/remote-machine-learning

## Files

- `immich_ml_modal.py` - Modal app with FastAPI auth proxy and Immich ML subprocess
- `Dockerfile.immich-modal` - Multi-stage Dockerfile: copies Immich ML into a CUDA runtime image with onnxruntime-gpu

## Related

- [immich-modal-proxy](https://github.com/zxzx1290/immich-modal-proxy) - Cloudflare Worker that sends authenticated requests to this endpoint

## Disclaimer

This is an unofficial community project and is not affiliated with or endorsed by the Immich team or Modal.

## License

MIT
