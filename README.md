# FitCheck VTON Engine

An exploration of how a virtual try-on model (person photo + garment photo → try-on image) could be served on **RunPod Serverless**. I built it while reading up on open VTON models (CatVTON, CatVTON-FLUX, Qwen-Image-Edit-2511) and RunPod's worker model, to understand the moving parts end to end: masking, preprocessing, diffusion, speed-ups and deployment.

It's an exploration, not a finished product. **So far everything has been tested on CPU only.** The diffusion engines have not yet been run on real GPU weights, so there are no real try-on results or GPU timings here yet. See [What's tested and what isn't](#whats-tested-and-what-isnt).

```
decode ─► body analysis (CPU) ─► photo check ─► 3:4 crop ─► mask ─► diffusion (GPU)
                                      │                                  │
                    reject unusable photos before using the GPU   paste result back into original ─► WebP
```

## What's in here

| Part | What it does |
|---|---|
| **Two diffusion engines** | `qwen_edit`: Qwen-Image-Edit-2511 with the Lightning 4-step LoRA and an FP8 transformer. Edits the photo from person + garment images, no mask needed as model input.<br>`catvton_flux`: CatVTON on FLUX.1-Fill-dev, reimplemented to load without text encoders.<br>`mock`: a CPU stand-in that paints the garment colour into the mask, for tests and smoke runs. |
| **Masking** | MediaPipe pose landmarks + multiclass selfie segmentation (both Apache-2.0), used to build `upper` / `lower` / `overall` masks without DensePose, SCHP or a detectron2 build. `overall` also covers shoes. Accepts a custom mask instead. |
| **Photo checks** | Rejects photos with no person, or with the needed body region out of frame, on CPU before any GPU time is used. Returns a stable error code the app could map to a message. |
| **Keeping the original photo** | Only pixels inside the feathered mask come from the model. The face, hair and background are pasted back from the original photo at full resolution. |
| **Speed-ups** | LoRA fused into the weights, FP8 via torchao, First-Block-Cache, torch.compile, optional Nunchaku INT4/FP4, and a text-free ONNX export of the FLUX transformer as the input for a TensorRT engine. Each is an environment variable. |
| **RunPod worker** | Docker image with exact version pins, weights baked in or loaded from RunPod cached models, a warm-up before the worker reports ready, per-step timings in every response, optional S3 output, and worker recycling after out-of-memory errors. |
| **Tooling** | Benchmark (p50/p95 per step, peak VRAM, $/try-on), endpoint client and load test, LoRA-merge bundler, model fetcher, mask gallery, CI workflow, RunPod Hub files. |

## Design notes

Things I ran into while exploring, and the choice each one led to:

- **CatVTON-FLUX doesn't use its text encoders.** It runs FLUX-Fill with no prompt, but loading the full pipeline downloads the whole repo (58.1 GB, per the Hub API). Loading only `transformer/`, `vae/` and `scheduler/` needs 24.2 GB, which matters for serverless cold starts.
- **The upstream FLUX try-on loop passes `encoder_hidden_states=None`.** Current diffusers sends that straight into a Linear layer and fails. Passing an empty (zero-length) text sequence gives the same "no text tokens" behaviour and works on current versions; [tests/test_catvton_flux.py](tests/test_catvton_flux.py) shows both.
- **Model licenses differ a lot.** The public CatVTON weights and FLUX.1-Fill-dev are non-commercial; Qwen-Image-Edit-2511 and MediaPipe are Apache-2.0. That's why there are two engines, and why the masker uses MediaPipe. (Summary table at the end; not legal advice.)
- **Qwen-Image-Edit-2511 is large.** bf16 weights are about 41 GB for the transformer plus 17 GB for the text encoder. The engine quantizes the transformer to FP8 on the GPU before moving the text encoder over, so the peak should fit a 48 GB card (not yet confirmed on hardware).
- **TensorRT and zero-length tensors don't mix well.** The ONNX export wraps the fused transformer so the empty text stream is built inside the graph, leaving four fixed-size inputs. onnxruntime matches PyTorch to 2.7e-5 on a tiny model.
- **CPU steps add up.** On a laptop the non-GPU work first took 1.59 s per request. Moving feathering to OpenCV, finding the garment on a thumbnail and running pose and segmentation in parallel brought it to 0.37 s.
- **RunPod specifics.** Cached models allow one Hugging Face repo per endpoint, so [scripts/build_bundle.py](scripts/build_bundle.py) merges the Lightning LoRA into a single repo. The SDK logs full outputs at DEBUG level, so the image sets `RUNPOD_DEBUG_LEVEL=INFO` to avoid logging whole base64 images.

## What's tested and what isn't

**Tested on CPU** (24 pytest tests, plus the scripts below):
- The CatVTON-FLUX loop runs end to end on a tiny, randomly initialised FLUX-Fill model with the real architecture: output shape, same seed gives the same image, cache state resets between requests, LoRA fuses into plain `Linear` layers.
- The ONNX export of the text-free FLUX transformer matches PyTorch.
- Masks and photo checks on four sample photos; face and hair pixels come back unchanged after compositing.
- Input handling: URLs, base64, data URIs, EXIF rotation, transparency, bad input.
- The RunPod handler completes a job through the SDK's local runner.

**Not tested yet:**
- Either engine on the real model weights, so no real try-on quality or GPU latency.
- FP8, torch.compile, Nunchaku and the TensorRT engine build on a GPU.
- The Docker image build and the CI workflow.
- Masks on a wide range of real user photos (only four sample photos so far).

## API

`POST https://api.runpod.ai/v2/<endpoint>/runsync`

```jsonc
{ "input": {
  "person_image":  "https://… | data:image/…;base64,… | <base64>",
  "garment_image": "https://… | …",
  "category": "upper",            // upper | lower | overall
  "mask_image": null,             // optional custom mask (white = regenerate), for in-app mask tools
  "seed": 42, "steps": null, "guidance": null,   // null = engine defaults
  "identity_lock": true,
  "output_format": "webp", "output_quality": 90, "max_output_side": 1536,
  "return_mask": false
}}
```

Response `output`:

```jsonc
{ "image": "data:image/webp;base64,…",   // or an S3 URL when BUCKET_* env is set
  "width": 960, "height": 1280, "seed": 42, "category": "upper",
  "engine": "qwen_edit", "commercial_ok": true,
  "warnings": ["legs_partially_visible"],
  "timings_ms": {"decode": …, "analyze": …, "mask": …, "preprocess": …, "diffusion": …, "compose": …, "encode": …, "total": …},
  "boot_ms": … }
```

Errors come back as the job `error` string `"<code>: <message>"`, with these codes:

| code | cause | suggested app UX |
|---|---|---|
| `no_person_detected` | no body found | "We couldn't find you in this photo" |
| `person_cut_off` | shoulders or hips not in frame | "Step back so we can see from shoulders to hips" |
| `invalid_input` / `invalid_image` / `image_too_large` / `fetch_failed` | bad request | client-side |
| `empty_mask` | nothing to replace for the chosen category | pick another category |
| `internal_error` | server-side fault | retry |

Warnings don't fail the job: `multiple_people`, `legs_partially_visible`.

## Deploy on RunPod

1. **Build the image.** Either push to `main` (the CI job builds `ghcr.io/<repo>`), or use RunPod's GitHub integration, or:
   ```bash
   docker build -t <registry>/fitcheck-vton:qwen --build-arg VTON_ENGINE=qwen_edit .
   # self-contained image with weights baked in:
   docker build ... --build-arg BAKE_MODELS=1 --secret id=hf_token,env=HF_TOKEN .
   ```
2. **Choose where the weights live.**
   - Baked into the image: most predictable cold start.
   - RunPod **cached model**: download isn't billed, but it's one repo per endpoint. Run `scripts/build_bundle.py` to merge Lightning into a single private repo, then set `QWEN_EDIT_REPO=<bundle>` and `QWEN_LIGHTNING_FUSED=1`.
3. **Create the endpoint.**
   - GPU: **L40S 48 GB** (FP8, 48 GB fits Qwen with the FP8 transformer) or H100 80 GB.
   - Workers: `max workers ≥ peak RPS × latency`, `active workers = 1` during business hours, idle timeout 5–15 s, FlashBoot on.
   - Container disk: at least 90 GB if baking.
4. **Smoke test.** `python scripts/client.py --person me.jpg --garment tee.jpg`
5. **Benchmark every preset** on the real GPU:
   ```bash
   VTON_FP8=1                python scripts/bench.py --engine qwen_edit --gpu-price-hr 1.75 --out out/qwen_fp8.json
   VTON_FP8=1 VTON_COMPILE=1 python scripts/bench.py --engine qwen_edit --gpu-price-hr 1.75 --out out/qwen_fp8_compile.json
   ```

### Tuning knobs (endpoint env vars)

| var | default | effect |
|---|---|---|
| `VTON_ENGINE` | `qwen_edit` | `qwen_edit` / `catvton_flux` / `mock` |
| `VTON_FP8` | `1` | torchao FP8 on the DiT; skipped automatically below sm_89 |
| `VTON_FBCACHE` | `0` | First-Block-Cache threshold. Useful at 20+ steps, pointless at 4 |
| `VTON_COMPILE` | `0` | `torch.compile`: faster steady state, adds boot time |
| `VTON_TRT_ENGINE` | – | path to a `.plan` built with `scripts/export_onnx.py` + `trtexec` (CatVTON-FLUX) |
| `VTON_WIDTH` / `VTON_HEIGHT` | 768 / 1024 | working canvas |
| `VTON_WARMUP` | `1` | one throwaway inference at boot |
| `QWEN_NUNCHAKU_TRANSFORMER` | – | SVDQuant INT4/FP4 transformer (image built with `INSTALL_NUNCHAKU=1`) |
| `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` | – | return S3/R2 URLs instead of base64 |

### TensorRT (CatVTON-FLUX DiT)

```bash
python scripts/export_onnx.py --out /workspace/onnx/catvton_flux.onnx          # fused LoRA, 4 static inputs
trtexec --onnx=/workspace/onnx/catvton_flux.onnx --bf16 --saveEngine=/workspace/catvton_flux.plan
VTON_ENGINE=catvton_flux VTON_TRT_ENGINE=/workspace/catvton_flux.plan python scripts/bench.py ...
```

Build the engine on the **same GPU type** the endpoint runs on; engines are tied to the GPU architecture. RunPod's GitHub builder has no GPU.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
python scripts/fetch_assets.py          # MediaPipe models + demo photos (demo photos are CC BY-NC-SA, local use only)
pytest -q
python scripts/mask_gallery.py          # out/mask_gallery.jpg
VTON_ENGINE=mock MEDIAPIPE_DIR=.assets/models python handler.py --test_input "$(cat test_input.json)"
```

```
handler.py                 RunPod entrypoint (load + warm-up at import, error mapping, S3 upload)
src/vton/service.py        request orchestration + validation
src/vton/masking.py        MediaPipe masker + QC gate
src/vton/preprocess.py     smart crop / garment trim (crop box kept for paste-back)
src/vton/compose.py        identity lock
src/vton/engines/          qwen_edit · catvton_flux · mock
src/vton/accel.py          LoRA fuse · FP8 · FBCache · compile
src/vton/trt.py            text-free FLUX wrapper + TensorRT runtime
scripts/                   bench · client · export_onnx · build_bundle · fetch_models · fetch_assets · mask_gallery
```

## Model licenses (from public model cards, not legal advice)

| component | license | commercial use? |
|---|---|---|
| Qwen-Image-Edit-2511 | Apache-2.0 | yes |
| lightx2v Qwen-Image-Edit-2511-Lightning | check the model card before launch | verify |
| MediaPipe pose + selfie multiclass | Apache-2.0 | yes |
| FLUX.1-Fill-dev | FLUX.1 [dev] Non-Commercial | only with a BFL commercial license |
| CatVTON weights (incl. flux-lora) | CC BY-NC-SA 4.0 | no |
| DensePose / SCHP checkpoints | trained on non-commercial datasets | risky |
