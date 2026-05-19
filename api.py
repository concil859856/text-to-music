"""
ACE-Step FastAPI Backend — supports all features:
  text2music, audio2audio, retake, repaint, edit, extend
"""

import os
import uuid
import time
import shutil

import click
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8187)
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