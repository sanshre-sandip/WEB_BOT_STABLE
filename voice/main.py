"""Voice assistant MVP: mic -> STT -> RAG backend -> TTS."""

from __future__ import annotations

import os
import sys
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parent
ROOT_DIR = VOICE_DIR.parent
for path in (ROOT_DIR, VOICE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rag_client import (  # noqa: E402
    DEFAULT_USE_RAG,
    check_backend,
    get_backend_url,
    query_backend,
    warmup_ollama,
)
from stt import preload_model, record, transcribe  # noqa: E402
from tts import speak  # noqa: E402

EXIT_PHRASES = {"quit", "exit", "goodbye", "stop"}


def main() -> None:
    record_seconds = float(os.getenv("VOICE_RECORD_SECONDS", "5"))
    whisper_model = os.getenv("WHISPER_MODEL", "tiny")
    tts_voice = os.getenv("TTS_VOICE", "en-US-JennyNeural")

    print("Voice assistant (MVP)")
    print(f"Backend: {get_backend_url()}")
    print(f"RAG: {'on' if DEFAULT_USE_RAG else 'off'}")
    print(f"Whisper model: {whisper_model}")

    if check_backend():
        print("Backend: reachable")
    else:
        print(
            "Backend: NOT reachable — start it with:\n"
            "  uvicorn RAG.main:app --host 0.0.0.0 --port 8011"
        )

    try:
        preload_model(model_size=whisper_model)
    except Exception as exc:
        print(f"Failed to load Whisper model: {exc}")
        sys.exit(1)

    warmup_ollama()

    print("Press Enter to record a question, or Ctrl+C to quit.\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            input(">> Press Enter to speak... ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        try:
            audio, sample_rate = record(record_seconds)
            text = transcribe(audio, sample_rate, model_size=whisper_model)
        except Exception as exc:
            print(f"STT error: {exc}")
            continue

        if not text:
            print("No speech detected. Try again.")
            continue

        print(f"You: {text}", flush=True)

        if text.lower().strip().rstrip(".!") in EXIT_PHRASES:
            speak("Goodbye.", voice=tts_voice)
            break

        print("Waiting for answer from RAG backend...", flush=True)
        try:
            answer = query_backend(text, history, retries=1)
        except Exception as exc:
            print(f"RAG error: {exc}")
            speak("Sorry, I could not reach the assistant.", voice=tts_voice)
            continue

        if not answer.strip():
            print("Bot returned an empty answer.")
            speak("Sorry, I did not get an answer.", voice=tts_voice)
            continue

        print(f"Bot: {answer}", flush=True)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})

        try:
            speak(answer, voice=tts_voice)
        except Exception as exc:
            print(f"TTS error: {exc}")
            print("The text answer is shown above even if audio playback failed.")


if __name__ == "__main__":
    main()
