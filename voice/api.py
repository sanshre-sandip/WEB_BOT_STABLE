"""FastAPI voice endpoints: STT, RAG, TTS with timing."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

VOICE_DIR = Path(__file__).resolve().parent
ROOT_DIR = VOICE_DIR.parent
for _path in (ROOT_DIR, VOICE_DIR):
    _p = str(_path)
    if _p not in __import__("sys").path:
        __import__("sys").path.insert(0, _p)

from stt import preload_model, transcribe  # noqa: E402
from tts import _synthesize_bytes  # noqa: E402

router = APIRouter(prefix="/voice", tags=["voice"])

_whisper_model_size = os.getenv("WHISPER_MODEL", "base")
_whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
_whisper_compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
_whisper_vad = os.getenv("VOICE_VAD_FILTER", "false").lower() == "true"
_tts_voice = os.getenv("TTS_VOICE", "en-US-JennyNeural")
_tts_rate = os.getenv("TTS_RATE", "+8%")
_tts_pitch = os.getenv("TTS_PITCH", "+4Hz")


class VoiceChatRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio (WAV/MP3/etc.)")
    sample_rate: int = Field(default=16000, description="Sample rate of input audio")
    history: list[dict[str, str]] = Field(default_factory=list)
    source: str | None = None
    k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    whisper_model: str | None = None
    tts_voice: str | None = None
    tts_rate: str | None = None
    tts_pitch: str | None = None


class VoiceChatResponse(BaseModel):
    status: str
    transcription: str
    answer: str
    audio_base64: str | None = None
    timings: dict[str, float]
    model_used: str | None = None
    provider_used: str | None = None


class TranscribeRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    model_size: str | None = None


class TranscribeResponse(BaseModel):
    status: str
    text: str
    timings: dict[str, float]


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str | None = None
    rate: str | None = None
    pitch: str | None = None
    format: Literal["mp3", "base64"] = "base64"


class SynthesizeResponse(BaseModel):
    status: str
    audio_base64: str | None = None
    timings: dict[str, float]


class VoiceStatusResponse(BaseModel):
    status: str
    stt_ready: bool
    tts_ready: bool
    backend_url: str | None


def _decode_audio(audio_base64: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio and normalize to 16kHz mono using FFmpeg."""
    import subprocess
    
    # Strip data URL prefix if present
    if "," in audio_base64[:100]:
        audio_base64 = audio_base64.split(",")[1]
        
    raw_bytes = base64.b64decode(audio_base64)
    target_sr = 16000
    
    try:
        # Use FFmpeg to decode and resample any format to 16kHz mono f32le PCM
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:0",          # Input from stdin
                "-f", "f32le",           # Output format float32 little endian
                "-acodec", "pcm_f32le",
                "-ar", str(target_sr),   # Resample to 16000 Hz
                "-ac", "1",              # Mono
                "pipe:1"                 # Output to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = process.communicate(input=raw_bytes)
        
        if process.returncode != 0:
            from RAG.main import logger
            logger.error(f"FFmpeg error: {err.decode()}")
            raise Exception("FFmpeg failed to decode audio")
            
        data = np.frombuffer(out, dtype=np.float32)
        if len(data) == 0:
            raise Exception("FFmpeg returned empty audio data")
            
        return data, target_sr
        
    except Exception as exc:
        from RAG.main import logger
        logger.warning(f"FFmpeg decoding failed ({exc}), trying fallback...")
        
    # Fallback for simple WAV/MP3 if FFmpeg fails
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(sr)
    except Exception:
        pass
        
    raise HTTPException(
        status_code=400, 
        detail="Failed to decode audio. Ensure you are sending valid Base64 data (WAV, MP3, or WebM)."
    )


async def _query_backend_async(query: str, history: list[dict[str, str]], source: str | None, k: int, temperature: float):
    try:
        from rag_client import query_backend
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, lambda: query_backend(query, history, source=source, k=k, temperature=temperature, retries=1)
        )
        return answer, None
    except Exception as exc:
        return None, str(exc)


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(request: VoiceChatRequest):
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    model_size = request.whisper_model or _whisper_model_size
    tts_voice = request.tts_voice or _tts_voice
    tts_rate = request.tts_rate or _tts_rate
    tts_pitch = request.tts_pitch or _tts_pitch

    try:
        preload_model(model_size=model_size, device=_whisper_device, compute_type=_whisper_compute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {exc}")

    audio, sr = _decode_audio(request.audio_base64)
    rms = np.sqrt(np.mean(audio**2))
    from RAG.main import logger
    logger.info(f"Voice Chat: Received {len(audio)} samples at {sr}Hz. RMS Volume: {rms:.6f}")

    t1 = time.perf_counter()
    timings["audio_decode_s"] = round(t1 - t0, 4)

    try:
        text = transcribe(
            audio,
            sr,
            model_size=model_size,
            device=_whisper_device,
            compute_type=_whisper_compute,
            vad_filter=_whisper_vad,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT failed: {exc}")
    t2 = time.perf_counter()
    timings["stt_s"] = round(t2 - t1, 4)

    if not text.strip():
        logger.warning(f"No speech detected. RMS Volume was {rms:.6f}")
        raise HTTPException(status_code=400, detail=f"No speech detected. Audio volume level: {rms:.6f}")

    answer, rag_error = await _query_backend_async(
        query=text,
        history=request.history,
        source=request.source,
        k=request.k,
        temperature=request.temperature,
    )
    t3 = time.perf_counter()
    timings["rag_s"] = round(t3 - t2, 4)

    if rag_error:
        raise HTTPException(status_code=502, detail=f"RAG backend error: {rag_error}")
    if not answer or not answer.strip():
        answer = "I'm sorry, I didn't get a response from the assistant."

    audio_base64_out = None
    tts_error = None
    try:
        audio_bytes = await _synthesize_bytes(answer.strip(), tts_voice, rate=tts_rate, pitch=tts_pitch)
        audio_base64_out = base64.b64encode(audio_bytes).decode()
    except Exception as exc:
        tts_error = str(exc)
    t4 = time.perf_counter()
    timings["tts_s"] = round(t4 - t3, 4)
    timings["total_s"] = round(t4 - t0, 4)

    status = "success" if audio_base64_out is not None else "partial"
    if tts_error:
        status = "tts_failed"

    return VoiceChatResponse(
        status=status,
        transcription=text,
        answer=answer,
        audio_base64=audio_base64_out,
        timings=timings,
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: TranscribeRequest):
    t0 = time.perf_counter()
    model_size = request.model_size or _whisper_model_size
    try:
        preload_model(model_size=model_size, device=_whisper_device, compute_type=_whisper_compute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {exc}")

    audio, sr = _decode_audio(request.audio_base64)
    t1 = time.perf_counter()
    timings: dict[str, float] = {"audio_decode_s": round(t1 - t0, 4)}

    try:
        text = transcribe(
            audio,
            sr,
            model_size=model_size,
            device=_whisper_device,
            compute_type=_whisper_compute,
            vad_filter=_whisper_vad,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT failed: {exc}")
    t2 = time.perf_counter()
    timings["stt_s"] = round(t2 - t1, 4)

    return TranscribeResponse(status="success", text=text, timings=timings)


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_text(request: SynthesizeRequest):
    t0 = time.perf_counter()
    voice = request.voice or _tts_voice
    rate = request.rate or _tts_rate
    pitch = request.pitch or _tts_pitch

    audio_base64_out = None
    tts_error = None
    try:
        import edge_tts
        communicate = edge_tts.Communicate(request.text, voice, rate=rate, pitch=pitch)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        audio_base64_out = base64.b64encode(bytes(buf)).decode()
    except Exception as exc:
        tts_error = str(exc)
    t1 = time.perf_counter()
    timings = {"tts_s": round(t1 - t0, 4)}

    if tts_error:
        raise HTTPException(status_code=500, detail=f"TTS failed: {tts_error}")

    return SynthesizeResponse(status="success", audio_base64=audio_base64_out, timings=timings)


@router.get("/status", response_model=VoiceStatusResponse)
async def voice_status():
    try:
        preload_model(model_size=_whisper_model_size, device=_whisper_device, compute_type=_whisper_compute)
        stt_ready = True
    except Exception:
        stt_ready = False

    try:
        tts_ready = True
    except Exception:
        tts_ready = False

    backend_url = None
    try:
        from rag_client import get_backend_url
        backend_url = get_backend_url()
    except Exception:
        pass

    return VoiceStatusResponse(
        status="ok",
        stt_ready=stt_ready,
        tts_ready=tts_ready,
        backend_url=backend_url,
    )
