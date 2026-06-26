"""Speech-to-text: record microphone audio and transcribe with faster-whisper."""

from __future__ import annotations

import numpy as np

try:
    import sounddevice as sd
except (OSError, ImportError) as exc:
    sd = None
    _import_error = exc

_model = None
_model_config: tuple[str, str, str] | None = None


def _get_model(model_size: str, device: str, compute_type: str):
    global _model, _model_config
    config = (model_size, device, compute_type)
    if _model is None or _model_config != config:
        from faster_whisper import WhisperModel

        print(
            f"Loading Whisper model '{model_size}' ({device}, {compute_type})...",
            "First run downloads from Hugging Face and can take a few minutes.",
            sep="\n  ",
            flush=True,
        )
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_config = config
        print("Whisper model ready.", flush=True)
    return _model


def preload_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    """Load the Whisper model up front so the first recording feels responsive."""
    _get_model(model_size, device, compute_type)


def record(duration_sec: float = 5.0, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Record mono audio from the default microphone."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is not available. On Ubuntu/Debian run:\n"
            "  sudo apt install portaudio19-dev libportaudio2"
        )
    print(f"Recording for {duration_sec:.0f}s... (speak now)", flush=True)
    frames = int(duration_sec * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.", flush=True)
    return audio.flatten(), sample_rate


def transcribe(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "en",
    vad_filter: bool = True,
) -> str:
    """Transcribe a float32 mono waveform with faster-whisper."""
    if sample_rate != 16000:
        print(f"Note: resampling from {sample_rate} Hz is not implemented; using as-is.", flush=True)

    model = _get_model(model_size, device, compute_type)
    print(f"Transcribing (VAD={vad_filter})...", flush=True)
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=1,
        vad_filter=vad_filter,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    print(f"Transcription done (detected language: {info.language}).", flush=True)
    return text
