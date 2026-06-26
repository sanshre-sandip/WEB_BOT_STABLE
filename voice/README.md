# Voice Assistant (MVP)

Local voice loop for the existing RAG chatbot:

**mic → faster-whisper (STT) → FastAPI `/generate/rag` → edge-tts (TTS) → speaker**

No streaming, no UI, no wake word. Press Enter, speak, hear the answer.

## Prerequisites

1. **FastAPI backend running** with the vector store loaded (same server the Streamlit chat uses):

   ```bash
   uvicorn RAG.main:app --host 0.0.0.0 --port 8000
   ```

2. **System audio**: a working microphone and speakers. On Linux you may need PortAudio:

   ```bash
   sudo apt install portaudio19-dev libportaudio2
   ```

3. **Python 3.10+** recommended.

4. **First run**: the Whisper model downloads from Hugging Face on startup. This can take several minutes on a slow connection. Use `WHISPER_MODEL=tiny` for faster CPU transcription.

## Install

From the project root:

```bash
pip install -r voice/requirements.txt
```

The first run downloads the Whisper model (size depends on `WHISPER_MODEL`, default `base`).

## Run

```bash
python voice/main.py
```

1. Press **Enter** to start a 5-second recording.
2. Your speech is transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. Text is sent to the RAG backend: `POST {BACKEND}/generate/rag` (same contract as `streamlight/chat.py`).
4. The reply is spoken with [edge-tts](https://github.com/rany2/edge-tts) (free neural voices).

Say **quit**, **exit**, or **goodbye** to end the session.

## Configuration

Uses the root `.env` (via `RAG/config.py`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `BACKEND` | `http://localhost:8000` | FastAPI base URL |
| `DEFAULT_USE_RAG` | `true` | `true` → `/generate/rag`, `false` → `/generate` |
| `DEFAULT_RAG_K` | `5` | Retrieval top-k |
| `DEFAULT_RAG_SOURCE_FILTER` | *(empty)* | Optional source filter |
| `DEFAULT_LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_PROVIDER` | `openai` | LLM provider passed to backend |
| `DEFAULT_LLM_MODEL` | *(empty)* | Optional model override |
| `CLIENT_REQUEST_TIMEOUT` | `600` | HTTP timeout (seconds) |

Voice-specific (optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VOICE_RECORD_SECONDS` | `5` | Recording length per turn |
| `WHISPER_MODEL` | `tiny` | faster-whisper model (`tiny`, `base`, `small`, …) |
| `TTS_VOICE` | `en-US-JennyNeural` | Young-adult female voice (`en-US-AvaNeural`, `en-US-EmmaNeural` also work) |
| `TTS_RATE` | `+8%` | Speech speed (`+0%` = normal) |
| `TTS_PITCH` | `+4Hz` | Voice pitch (`+0Hz` = normal) |

List voices: `edge-tts --list-voices`

## RAG integration

The assistant calls the **REST API**, not Python imports directly:

```
POST {BACKEND}/generate/rag
{
  "query": "<transcribed text>",
  "source": null,
  "k": 5,
  "history": [{"role": "user", "content": "..."}, ...],
  "temperature": 0.2,
  "stream": false
}
```

Response field used: `answer`.

## Files

| File | Role |
|------|------|
| `stt.py` | `sounddevice` recording + `faster-whisper` transcription |
| `tts.py` | `edge-tts` synthesis + `pygame` playback |
| `main.py` | Record → transcribe → RAG → speak loop |
| `requirements.txt` | Voice-only dependencies |

## Troubleshooting

- **Backend connection refused** — start `uvicorn RAG.main:app` and check `BACKEND` in `.env`.
- **No microphone** — verify with `python -c "import sounddevice; print(sounddevice.query_devices())"`.
- **Slow first transcription** — Whisper model loads once; use `WHISPER_MODEL=tiny` for faster CPU inference.
- **TTS silent / error** — ensure speakers work; try another `TTS_VOICE`.
