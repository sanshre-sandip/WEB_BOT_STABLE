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


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample a mono float32 waveform to a target sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    num_samples = int(round(len(audio) * target_sr / orig_sr))
    if num_samples <= 0:
        raise ValueError(f"Invalid resampling length: {num_samples}")

    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, num_samples)
    resampled = np.interp(new_indices, old_indices, audio).astype(np.float32)
    return resampled


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
        print(f"Resampling from {sample_rate} Hz to 16000 Hz...", flush=True)
        audio = _resample_audio(audio, sample_rate, target_sr=16000)
        sample_rate = 16000

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
