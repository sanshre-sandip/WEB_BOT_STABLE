"""Text-to-speech via edge-tts (free Microsoft neural voices)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile

# Warm, youthful US English female — good fit for a young adult assistant.
DEFAULT_VOICE = "en-US-JennyNeural"
DEFAULT_RATE = "+8%"
DEFAULT_PITCH = "+4Hz"


async def _synthesize(
    text: str,
    voice: str,
    output_path: str,
    *,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


async def _synthesize_bytes(
    text: str,
    voice: str,
    *,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


def _play_with_command(path: str, command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _play_mp3(path: str) -> None:
    players: list[tuple[str, list[str]]] = []

    if shutil.which("ffplay"):
        players.append(("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]))
    if shutil.which("mpv"):
        players.append(("mpv", ["mpv", "--no-video", "--really-quiet", path]))
    if shutil.which("pw-play"):
        players.append(("pw-play", ["pw-play", path]))
    if shutil.which("mpg123"):
        players.append(("mpg123", ["mpg123", "-q", path]))

    for _name, command in players:
        try:
            _play_with_command(path, command)
            return
        except (OSError, subprocess.CalledProcessError):
            continue

    import pygame

    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)


def speak(
    text: str,
    voice: str | None = None,
    *,
    rate: str | None = None,
    pitch: str | None = None,
    max_chars: int = 1500,
) -> None:
    """Synthesize *text* and play it through the default speaker."""
    if not text.strip():
        return

    voice = voice or os.getenv("TTS_VOICE", DEFAULT_VOICE)
    rate = rate or os.getenv("TTS_RATE", DEFAULT_RATE)
    pitch = pitch or os.getenv("TTS_PITCH", DEFAULT_PITCH)

    spoken_text = text.strip()
    if len(spoken_text) > max_chars:
        spoken_text = spoken_text[:max_chars].rsplit(" ", 1)[0] + "..."

    print("Speaking response...", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    try:
        asyncio.run(_synthesize(spoken_text, voice, path, rate=rate, pitch=pitch))
        _play_mp3(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print("Done speaking.", flush=True)
