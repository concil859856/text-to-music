"""
ACE-Step FastAPI Backend — supports all features:
  text2music, audio2audio, retake, repaint, edit, extend

Vocence /studio/ops integration: also exposes /healthz + /metrics with
bearer auth (via MUSIC_API_KEY env) so the fleet dashboard can scrape
status + per-pod traffic counters. Legacy /health endpoint preserved.
"""

import asyncio
import collections
import os
import threading
import time as _time
import uuid
import time
import shutil

import click
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from acestep.pipeline_ace_step import ACEStepPipeline

# ---------------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------------
app = FastAPI(title="ACE-Step API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.environ.get("ACE_OUTPUT_DIR", "./api_outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

pipeline: Optional[ACEStepPipeline] = None


# ---------------------------------------------------------------------------
# Vocence /studio/ops integration — env-driven config + bearer-auth'd
# /healthz + /metrics + inflight middleware.
# ---------------------------------------------------------------------------
MUSIC_API_KEY = (os.environ.get("MUSIC_API_KEY") or "").strip()
# Music gen is heavy (~30-60s per request). cap=1 so concurrent callers
# fail fast with 503 rather than wait minutes in queue.
MUSIC_CAP = int(os.environ.get("MUSIC_CAP") or "1")


class _Metrics:
    def __init__(self, recent_window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.start_ts = _time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_err: dict[str, int] = {}
        self.duration_ms_sum = 0.0
        self.duration_ms_count = 0
        self.recent_durations_ms: "collections.deque[float]" = collections.deque(maxlen=recent_window)
        self.bytes_sent_total = 0
        self.audio_ms_total = 0

    def record_success(self, duration_ms: float, bytes_sent: int = 0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.duration_ms_sum += duration_ms
            self.duration_ms_count += 1
            self.recent_durations_ms.append(duration_ms)
            self.bytes_sent_total += bytes_sent

    def record_error(self, code: str, duration_ms: float = 0.0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_err[code] = self.requests_err.get(code, 0) + 1
            if duration_ms > 0:
                self.duration_ms_sum += duration_ms
                self.duration_ms_count += 1
                self.recent_durations_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            durations = sorted(self.recent_durations_ms)
            n = len(durations)
            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                return durations[min(n - 1, int(p * n))]
            return {
                "uptime_seconds": int(_time.time() - self.start_ts),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_err": dict(self.requests_err),
                "duration_ms_sum": self.duration_ms_sum,
                "duration_ms_count": self.duration_ms_count,
                "duration_ms_avg": (self.duration_ms_sum / self.duration_ms_count) if self.duration_ms_count else 0.0,
                "duration_ms_p50": pct(0.50),
                "duration_ms_p95": pct(0.95),
                "duration_ms_p99": pct(0.99),
                "bytes_sent_total": self.bytes_sent_total,
                "audio_ms_total": self.audio_ms_total,
            }


_metrics = _Metrics()


class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = max(1, int(cap))
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int: return self._cap

    @property
    def inflight(self) -> int: return self._count

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)


_inflight = _InflightTracker(cap=MUSIC_CAP)

# Endpoints that bypass the inflight + metrics middleware.
_OPS_PATHS = {"/healthz", "/metrics", "/health", "/docs", "/redoc", "/openapi.json"}


def _check_bearer(request: Request) -> Optional[JSONResponse]:
    if not MUSIC_API_KEY:
        return None
    if request.headers.get("authorization", "") == f"Bearer {MUSIC_API_KEY}":
        return None
    return JSONResponse(
        {"type": "error", "code": "auth", "message": "missing or invalid bearer token"},
        status_code=401,
    )


@app.middleware("http")
async def _ops_middleware(request: Request, call_next):
    path = request.url.path
    # The /audio/{filename} download path also bypasses the inflight check —
    # it's a static file serve, not a synthesis request.
    if path in _OPS_PATHS or path.startswith("/audio/"):
        return await call_next(request)

    if not await _inflight.try_acquire():
        _metrics.record_error("server_busy", 0)
        return JSONResponse(
            {"type": "error", "code": "server_busy", "message": f"inflight cap ({_inflight.cap}) exhausted"},
            status_code=503,
        )

    t0 = _time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (_time.perf_counter() - t0) * 1000.0
        if response.status_code < 400:
            _metrics.record_success(elapsed)
        else:
            _metrics.record_error(f"http_{response.status_code}", elapsed)
        return response
    except Exception as e:
        _metrics.record_error(f"exception_{type(e).__name__}", (_time.perf_counter() - t0) * 1000.0)
        raise
    finally:
        await _inflight.release()


@app.get("/healthz")
async def healthz(request: Request):
    """Vocence ops dashboard health probe."""
    err = _check_bearer(request)
    if err is not None:
        return err
    return JSONResponse({
        "status": "ok",
        "service": "music",
        "model_id": "ACE-Step/ACE-Step-v1-3.5B",
        "sample_rate": 44100,
        "inflight": _inflight.inflight,
        "cap": _inflight.cap,
        "loaded": pipeline is not None and getattr(pipeline, "loaded", False),
        "dev_stub": False,
    })


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    """Vocence ops dashboard scrape endpoint."""
    err = _check_bearer(request)
    if err is not None:
        return err
    snap = _metrics.snapshot()
    snap["service"] = "music"
    snap["inflight"] = _inflight.inflight
    snap["cap"] = _inflight.cap
    return JSONResponse(snap)


def get_pipeline() -> ACEStepPipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_seeds(raw: Optional[str]) -> Optional[List[int]]:
    if not raw or raw.strip() == "":
        return None
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def parse_oss_steps(raw: Optional[str]) -> Optional[str]:
    if not raw or raw.strip() == "":
        return ""
    return raw.strip()


def save_upload(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "audio.wav")[1] or ".wav"
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return path


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _cleanup_generated(audio_path: str) -> None:
    """Delete the served audio file and its sidecar _input_params.json."""
    _safe_remove(audio_path)
    base, _ = os.path.splitext(audio_path)
    _safe_remove(f"{base}_input_params.json")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None and pipeline.loaded}


# ---- Text2Music ----
@app.post("/generate/text2music")
async def text2music(
    prompt: str = Form(...),
    lyrics: str = Form(""),
    audio_duration: float = Form(60.0),
    format: str = Form("wav"),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            task="text2music",
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Audio2Audio ----
@app.post("/generate/audio2audio")
async def audio2audio(
    ref_audio: UploadFile = File(...),
    prompt: str = Form(...),
    lyrics: str = Form(""),
    audio_duration: float = Form(60.0),
    format: str = Form("wav"),
    ref_audio_strength: float = Form(0.5),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    ref_path = save_upload(ref_audio)
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            audio2audio_enable=True,
            ref_audio_input=ref_path,
            ref_audio_strength=ref_audio_strength,
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            task="text2music",
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_remove(ref_path)


# ---- Retake ----
@app.post("/generate/retake")
async def retake(
    src_audio: UploadFile = File(...),
    prompt: str = Form(...),
    lyrics: str = Form(""),
    format: str = Form("wav"),
    retake_variance: float = Form(0.2),
    retake_seeds: str = Form(""),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    src_path = save_upload(src_audio)
    import librosa
    audio_duration = librosa.get_duration(filename=src_path)
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            retake_seeds=parse_seeds(retake_seeds),
            retake_variance=retake_variance,
            task="retake",
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_remove(src_path)


# ---- Repaint ----
@app.post("/generate/repaint")
async def repaint(
    src_audio: UploadFile = File(...),
    prompt: str = Form(...),
    lyrics: str = Form(""),
    format: str = Form("wav"),
    repaint_start: float = Form(0.0),
    repaint_end: float = Form(30.0),
    retake_variance: float = Form(0.2),
    retake_seeds: str = Form(""),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    src_path = save_upload(src_audio)
    import librosa
    audio_duration = librosa.get_duration(filename=src_path)
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            retake_seeds=parse_seeds(retake_seeds),
            retake_variance=retake_variance,
            task="repaint",
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_audio_path=src_path,
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_remove(src_path)


# ---- Edit ----
@app.post("/generate/edit")
async def edit(
    src_audio: UploadFile = File(...),
    prompt: str = Form(...),
    lyrics: str = Form(""),
    edit_target_prompt: str = Form(...),
    edit_target_lyrics: str = Form(""),
    format: str = Form("wav"),
    edit_n_min: float = Form(0.6),
    edit_n_max: float = Form(1.0),
    edit_n_avg: int = Form(1),
    retake_seeds: str = Form(""),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    src_path = save_upload(src_audio)
    import librosa
    audio_duration = librosa.get_duration(filename=src_path)
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            task="edit",
            src_audio_path=src_path,
            edit_target_prompt=edit_target_prompt,
            edit_target_lyrics=edit_target_lyrics,
            edit_n_min=edit_n_min,
            edit_n_max=edit_n_max,
            edit_n_avg=edit_n_avg,
            retake_seeds=parse_seeds(retake_seeds),
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_remove(src_path)


# ---- Extend ----
@app.post("/generate/extend")
async def extend(
    src_audio: UploadFile = File(...),
    prompt: str = Form(...),
    lyrics: str = Form(""),
    format: str = Form("wav"),
    left_extend_length: float = Form(0.0),
    right_extend_length: float = Form(30.0),
    extend_seeds: str = Form(""),
    infer_step: int = Form(60),
    guidance_scale: float = Form(15.0),
    scheduler_type: str = Form("euler"),
    cfg_type: str = Form("apg"),
    omega_scale: float = Form(10.0),
    manual_seeds: str = Form(""),
    guidance_interval: float = Form(0.5),
    guidance_interval_decay: float = Form(0.0),
    min_guidance_scale: float = Form(3.0),
    use_erg_tag: bool = Form(True),
    use_erg_lyric: bool = Form(False),
    use_erg_diffusion: bool = Form(True),
    oss_steps: str = Form(""),
    guidance_scale_text: float = Form(0.0),
    guidance_scale_lyric: float = Form(0.0),
    lora_name_or_path: str = Form("none"),
    lora_weight: float = Form(1.0),
):
    pipe = get_pipeline()
    src_path = save_upload(src_audio)
    import librosa
    audio_duration = librosa.get_duration(filename=src_path)
    repaint_start = -left_extend_length
    repaint_end = audio_duration + right_extend_length
    save_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{format}")
    try:
        result = pipe(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=parse_seeds(manual_seeds),
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=parse_oss_steps(oss_steps),
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            retake_seeds=parse_seeds(extend_seeds),
            retake_variance=1.0,
            task="extend",
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_audio_path=src_path,
            lora_name_or_path=lora_name_or_path,
            lora_weight=lora_weight,
            save_path=save_path,
        )
        params = result[-1]
        audio_path = result[0]
        return {"status": "success", "audio_path": audio_path, "params": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _safe_remove(src_path)


# ---- Serve generated audio files ----
@app.get("/audio/{filename:path}")
async def serve_audio(filename: str, background_tasks: BackgroundTasks):
    filepath = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(filepath):
        # try the raw path in case it's absolute from params
        filepath = filename
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    # Delete the file (and its sidecar params json) after the response is sent,
    # so the caller gets the bytes and we don't accumulate files on disk.
    background_tasks.add_task(_cleanup_generated, filepath)
    return FileResponse(filepath, media_type="audio/wav")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
@click.command()
@click.option("--checkpoint_path", type=str, default="", help="Model checkpoint directory")
@click.option("--host", type=str, default=os.environ.get("HOST") or "0.0.0.0")
# Default port 8115 matches the Vocence /studio/ops dashboard's music
# service convention. Existing pods on 8187 still work — pass --port=8187
# or set PORT=8187 in env. Click reads env vars when default lambda is used.
@click.option("--port", type=int, default=lambda: int(os.environ.get("PORT") or "8115"))
@click.option("--device_id", type=int, default=0)
@click.option("--bf16", type=bool, default=True)
@click.option("--torch_compile", type=bool, default=False)
@click.option("--cpu_offload", type=bool, default=False)
@click.option("--overlapped_decode", type=bool, default=False)
def main(checkpoint_path, host, port, device_id, bf16, torch_compile, cpu_offload, overlapped_decode):
    global pipeline
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    pipeline = ACEStepPipeline(
        checkpoint_dir=checkpoint_path or None,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()