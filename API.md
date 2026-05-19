# ACE-Step API

FastAPI backend exposing all six ACE-Step generation modes plus a file-serving endpoint with delivery-confirmed cleanup. Source: [api.py](api.py).

---

## Running the server

```bash
python api.py [--checkpoint_path PATH] [--host 0.0.0.0] [--port 8187] \
              [--device_id 0] [--bf16 true] [--torch_compile false] \
              [--cpu_offload false] [--overlapped_decode false]
```

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--checkpoint_path` | str | `""` | Model checkpoint directory. Empty falls back to the pipeline default. |
| `--host` | str | `0.0.0.0` | Bind address. |
| `--port` | int | `8187` | HTTP port. |
| `--device_id` | int | `0` | Sets `CUDA_VISIBLE_DEVICES`. |
| `--bf16` | bool | `true` | Pipeline dtype: `bfloat16` if true, else `float32`. |
| `--torch_compile` | bool | `false` | Enable `torch.compile`. |
| `--cpu_offload` | bool | `false` | Offload model layers to CPU. |
| `--overlapped_decode` | bool | `false` | Overlapped VAE decode. |

### Environment variables

| Var | Default | Purpose |
|---|---|---|
| `ACE_OUTPUT_DIR` | `./api_outputs` | Root directory for generated audio and uploaded reference files. |

The server auto-creates `$ACE_OUTPUT_DIR` and `$ACE_OUTPUT_DIR/uploads` on startup.

### CORS

CORS is wide open (`allow_origins=["*"]`, all methods, all headers). Lock this down before exposing the API externally.

---

## Content types & common patterns

- All `POST /generate/*` endpoints accept **`multipart/form-data`** (never JSON) because every variant except text2music takes a file upload, and even text2music uses `Form(...)` for consistency.
- All `POST /generate/*` endpoints return **`application/json`** with the shape:

  ```json
  {
    "status": "success",
    "audio_path": "./api_outputs/<timestamp>_<hex>.<format>",
    "params": { ... full pipeline params, including timecosts and actual seeds ... }
  }
  ```

- Errors return HTTP 500 with `{"detail": "<message>"}`. The pipeline raises 503 with `{"detail": "Pipeline not loaded yet"}` if requests arrive before model load completes.
- To retrieve the audio bytes, call `GET /audio/<basename>` exactly once — see [File-system lifecycle](#file-system-lifecycle).

### Shared form fields

These appear on every `POST /generate/*` endpoint with identical defaults. They're documented once here and not repeated per-endpoint.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | str | **required** | Style/genre tags. Plain text, comma-separated. |
| `lyrics` | str | `""` | Optional lyrics. Use `[verse]`, `[chorus]` section markers and `\n` for line breaks. |
| `format` | str | `"wav"` | Output container: `wav`, `mp3`, or `flac`. |
| `infer_step` | int | `60` | Diffusion steps. Lower = faster, lower quality. 27 is a common quality-vs-speed sweet spot. |
| `guidance_scale` | float | `15.0` | Classifier-free guidance scale. |
| `scheduler_type` | str | `"euler"` | Sampler. Other values: `heun`. |
| `cfg_type` | str | `"apg"` | CFG variant. |
| `omega_scale` | float | `10.0` | APG omega scale. |
| `manual_seeds` | str | `""` | Comma-separated ints, e.g. `"42,1234"`. Empty = random. |
| `guidance_interval` | float | `0.5` | |
| `guidance_interval_decay` | float | `0.0` | |
| `min_guidance_scale` | float | `3.0` | |
| `use_erg_tag` | bool | `true` | Entropy-regularized guidance on tags. |
| `use_erg_lyric` | bool | `false` | Entropy-regularized guidance on lyrics. |
| `use_erg_diffusion` | bool | `true` | Entropy-regularized guidance on diffusion. |
| `oss_steps` | str | `""` | Comma-separated step indices for one-step sampling. Empty = disabled. |
| `guidance_scale_text` | float | `0.0` | Per-channel guidance for text. |
| `guidance_scale_lyric` | float | `0.0` | Per-channel guidance for lyrics. |
| `lora_name_or_path` | str | `"none"` | LoRA adapter name or path. `"none"` disables. |
| `lora_weight` | float | `1.0` | LoRA strength. |

---

## Endpoints

### `GET /health`

Liveness + readiness probe.

**Response**

```json
{ "status": "healthy", "pipeline_loaded": true }
```

`pipeline_loaded` is `false` until the model finishes loading on startup.

---

### `POST /generate/text2music`

Pure text-to-music. No file inputs.

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `audio_duration` | float | `60.0` | Length of the generated clip in seconds. |

Plus all [shared fields](#shared-form-fields).

**Example**

```bash
curl -X POST http://localhost:8187/generate/text2music \
  -F "prompt=upbeat electronic dance, synth lead, 128bpm" \
  -F "lyrics=[verse]\nhello world\n[chorus]\ntesting one two three" \
  -F "audio_duration=20" \
  -F "infer_step=27" \
  -F "format=mp3"
```

---

### `POST /generate/audio2audio`

Style-transfer: generate a new clip conditioned on both the text prompt and a reference audio file.

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `ref_audio` | **file** | **required** | Reference audio (any container ffmpeg/librosa can read). |
| `ref_audio_strength` | float | `0.5` | How strongly the reference conditions the output. `0` ≈ text-only, `1` ≈ heavy conditioning. |
| `audio_duration` | float | `60.0` | Length of the generated clip in seconds (independent of the reference's duration). |

Plus all [shared fields](#shared-form-fields).

**Example**

```bash
curl -X POST http://localhost:8187/generate/audio2audio \
  -F "ref_audio=@/path/to/ref.mp3" \
  -F "prompt=jazz piano trio, walking bass, brushed drums" \
  -F "audio_duration=20" \
  -F "ref_audio_strength=0.5" \
  -F "infer_step=27" \
  -F "format=mp3"
```

---

### `POST /generate/retake`

Regenerate variations of a track. Retake does **not** condition on the source audio's latents — it regenerates from the same `prompt` + `lyrics` using new seeds tempered toward the original via `retake_variance`. The uploaded `src_audio` is used **only** to derive the output duration via `librosa.get_duration`.

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `src_audio` | **file** | **required** | Source track. Its duration becomes the output duration. |
| `retake_variance` | float | `0.2` | `0.0` = identical seeds (deterministic), `1.0` = fully fresh seeds. |
| `retake_seeds` | str | `""` | Comma-separated seeds for retake. Empty = random. |

Plus all [shared fields](#shared-form-fields) **except** `audio_duration` (auto-derived from `src_audio`).

**Example**

```bash
curl -X POST http://localhost:8187/generate/retake \
  -F "src_audio=@/path/to/original.mp3" \
  -F "prompt=upbeat electronic dance" \
  -F "retake_variance=0.3" \
  -F "infer_step=27" \
  -F "format=mp3"
```

---

### `POST /generate/repaint`

Inpaint a time window of an existing track. Frames outside `[repaint_start, repaint_end]` are preserved; frames inside are regenerated from `prompt` + `lyrics`.

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `src_audio` | **file** | **required** | Source track being repainted. |
| `repaint_start` | float | `0.0` | Start of the repaint window in seconds. |
| `repaint_end` | float | `30.0` | End of the repaint window in seconds. Must be ≤ source duration. |
| `retake_variance` | float | `0.2` | Variance of the noise injected into the repaint window. |
| `retake_seeds` | str | `""` | Seeds for the regenerated window. |

Plus all [shared fields](#shared-form-fields) **except** `audio_duration` (auto-derived from `src_audio`).

**Example**

```bash
curl -X POST http://localhost:8187/generate/repaint \
  -F "src_audio=@/path/to/original.mp3" \
  -F "prompt=upbeat electronic dance, synth lead, 128bpm" \
  -F "repaint_start=10" \
  -F "repaint_end=20" \
  -F "retake_variance=0.5" \
  -F "format=mp3"
```

---

### `POST /generate/edit`

Edit the style and/or lyrics of an existing track while preserving its rhythmic/structural shape.

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `src_audio` | **file** | **required** | Source track being edited. |
| `prompt` | str | **required** | Description of the **original** track. |
| `lyrics` | str | `""` | Original lyrics. |
| `edit_target_prompt` | str | **required** | Description of the **desired** result. |
| `edit_target_lyrics` | str | `""` | Desired lyrics. |
| `edit_n_min` | float | `0.6` | Lower bound of the noise schedule used for editing. |
| `edit_n_max` | float | `1.0` | Upper bound of the noise schedule used for editing. |
| `edit_n_avg` | int | `1` | Number of averaging passes. |
| `retake_seeds` | str | `""` | Seeds for the edit. |

Plus all [shared fields](#shared-form-fields) **except** `audio_duration` (auto-derived from `src_audio`).

**Example**

```bash
curl -X POST http://localhost:8187/generate/edit \
  -F "src_audio=@/path/to/original.mp3" \
  -F "prompt=upbeat electronic dance, synth lead, 128bpm" \
  -F "edit_target_prompt=mellow acoustic guitar, slow tempo, ambient" \
  -F "edit_n_min=0.6" \
  -F "edit_n_max=1.0" \
  -F "edit_n_avg=1" \
  -F "format=mp3"
```

---

### `POST /generate/extend`

Extend an existing track to the left and/or right. Internally implemented as a repaint where the repaint window extends beyond the original audio:

- `repaint_start = -left_extend_length`
- `repaint_end   = src_duration + right_extend_length`

`retake_variance` is forced to `1.0` (fully fresh latents in the extended regions).

**Endpoint-specific fields**

| Field | Type | Default | Notes |
|---|---|---|---|
| `src_audio` | **file** | **required** | Source track being extended. |
| `left_extend_length` | float | `0.0` | Seconds to prepend before the source. |
| `right_extend_length` | float | `30.0` | Seconds to append after the source. |
| `extend_seeds` | str | `""` | Seeds for the extension regions. |

Plus all [shared fields](#shared-form-fields) **except** `audio_duration` (auto-derived from `src_audio`).

**Example**

```bash
curl -X POST http://localhost:8187/generate/extend \
  -F "src_audio=@/path/to/original.mp3" \
  -F "prompt=upbeat electronic dance, synth lead, 128bpm" \
  -F "left_extend_length=0" \
  -F "right_extend_length=10" \
  -F "format=mp3"
```

---

### `GET /audio/{filename}`

Download a generated audio file. **The file is deleted from disk after the response finishes streaming** — see [File-system lifecycle](#file-system-lifecycle).

`{filename}` is the basename of `audio_path` returned by a `/generate/*` call (the leading `./api_outputs/` is stripped before lookup, so passing either form works).

**Response**

- `200`: audio bytes, `Content-Type: audio/wav` (header is `audio/wav` regardless of actual container — `mp3`/`flac` files still stream correctly; check the file extension or magic bytes for the true format).
- `404`: `{"detail": "Audio file not found"}` — typically because the file was already fetched once and cleaned up.

**Example**

```bash
curl -O http://localhost:8187/audio/1778766098_a57b6a2d.mp3
```

---

## File-system lifecycle

The API writes to two directories under `$ACE_OUTPUT_DIR` (default `./api_outputs`):

```
api_outputs/
├── <timestamp>_<hex>.<ext>                  # generated audio
├── <timestamp>_<hex>_input_params.json      # sidecar metadata (same basename as the audio)
└── uploads/
    └── <uuid>.<ext>                         # uploaded src_audio / ref_audio
```

### Generated audio + sidecar JSON

Lifetime: **created on `POST /generate/*`, deleted on `GET /audio/{filename}`**.

- The pipeline writes the audio file at the path it returns in `audio_path`.
- The pipeline also writes `<same-basename>_input_params.json` alongside it (full param dump used by the gradio UI for retake/etc. — the API doesn't need it, but it's still produced).
- When the file is served via `GET /audio/...`, FastAPI schedules a `BackgroundTasks` cleanup that fires **after the response body has been fully sent**. It removes both the audio file and its `_input_params.json` sidecar.
- **Single-fetch semantics.** A second `GET` for the same filename returns 404. If your client retries on a transient HTTP error, retry the whole `POST /generate/*` flow rather than re-`GET`-ing the same path.
- **If you never call `/audio`,** the file stays on disk indefinitely. There is no TTL sweep. Either always fetch, or run an out-of-band cleanup.

### Uploaded reference / source audio

Lifetime: **created on `POST /generate/{audio2audio,retake,repaint,edit,extend}`, deleted in a `finally` block once the pipeline call returns**.

- Each upload-taking endpoint copies the multipart upload to `uploads/<uuid>.<ext>` via `save_upload`.
- The pipeline reads the file inside the `pipe(...)` call. Once that returns (whether the request succeeds or raises), `_safe_remove` deletes the upload.
- This means uploads never persist across requests, even if the pipeline raises mid-generation.

### Cleanup helpers ([api.py:70-84](api.py#L70-L84))

```python
def _safe_remove(path):           # idempotent unlink, swallows OSError
def _cleanup_generated(audio_path):  # removes the audio and its _input_params.json sidecar
```

`_safe_remove` is called from each generate endpoint's `finally` block. `_cleanup_generated` is registered on `BackgroundTasks` inside `/audio/{filename}`.

---

## End-to-end client flow

```text
1. POST /generate/<mode>           (multipart, with files if applicable)
   └─> on return:
       • src_audio / ref_audio uploads have been deleted
       • response JSON contains audio_path + params (incl. actual_seeds, timecosts)

2. GET  /audio/<basename of audio_path>
   └─> on return:
       • client has the audio bytes
       • server has deleted the audio file and its _input_params.json sidecar
```

A complete cycle leaves nothing on disk except the running model weights.
