# Voice Chat Audio Encoding Fix

## Problem

The `/voice/chat` endpoint was returning a 400 error:
```
Failed to decode audio. Ensure you are sending valid Base64 data (WAV, MP3, or WebM).
```

## Root Causes

1. **No validation of input**: The `audio_base64` parameter wasn't validated before decoding
2. **Missing error handling**: The `base64.b64decode()` call had no try-except block, causing unhelpful error messages
3. **Missing dependency**: `soundfile` was not in `requirements.txt`, so the fallback decoder couldn't run
4. **Empty audio data**: No check for empty decoded bytes
5. **No timeout**: FFmpeg processing could hang indefinitely

## Changes Made

### 1. Enhanced Error Handling in `voice/api.py`

- ✅ Added validation for empty `audio_base64` strings
- ✅ Wrapped `base64.b64decode()` in try-except with descriptive error messages
- ✅ Added check for empty decoded audio data
- ✅ Added timeout to FFmpeg process (10 seconds)
- ✅ Better logging for debugging
- ✅ Proper error messages that indicate what went wrong

### 2. Updated Dependencies in `voice/requirements.txt`

Added:
- `soundfile` - Fallback audio decoder (WAV, FLAC, OGG, etc.)
- `ffmpeg-python` - Better FFmpeg integration

## How to Use the Voice Chat Endpoint

### Option 1: Use the Test Client Script

```bash
# Generate test audio and send to endpoint
python test_voice_client.py --generate --backend http://localhost:8011

# Or use an existing audio file
python test_voice_client.py --audio /path/to/audio.wav --backend http://localhost:8011
```

### Option 2: Manual cURL Request

First, encode your audio file to Base64:

```bash
# On Linux/macOS
AUDIO_BASE64=$(base64 < /path/to/audio.wav)

# Then send to the endpoint
curl -X POST "http://localhost:8011/voice/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"audio_base64\": \"$AUDIO_BASE64\",
    \"sample_rate\": 16000,
    \"history\": [],
    \"k\": 5,
    \"temperature\": 0.2
  }"
```

### Option 3: Python Code

```python
import base64
import httpx

# Read audio file
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

# Encode to Base64
audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

# Send request
response = httpx.post(
    "http://localhost:8011/voice/chat",
    json={
        "audio_base64": audio_base64,
        "sample_rate": 16000,
        "history": [],
        "k": 5,
        "temperature": 0.2,
    },
    timeout=60,
)

print(response.json())
```

## Audio Format Requirements

The endpoint accepts any audio format that FFmpeg or soundfile can handle:

- ✅ WAV (Waveform Audio File)
- ✅ MP3 (MPEG Audio)
- ✅ WebM (Web Media)
- ✅ OGG (Ogg Vorbis)
- ✅ FLAC (Free Lossless Audio Codec)
- ✅ And many more...

### Recommended Format

**WAV with these specifications:**
- Sample rate: 16000 Hz (mono or stereo, will be converted to mono)
- Bit depth: 16-bit or higher
- Channels: Mono or stereo

Example generating test audio with Python:

```python
import numpy as np
import soundfile as sf

# Generate 3 seconds of silence (or speech if recording)
duration = 3.0
sample_rate = 16000
num_samples = int(duration * sample_rate)

# Silence
audio_data = np.zeros(num_samples, dtype=np.float32)

# Save as WAV
sf.write("audio.wav", audio_data, sample_rate)

# Or record from microphone
from sounddevice import rec
audio_data = rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sf.write("audio.wav", audio_data, sample_rate)
```

## Request Parameters

```json
{
  "audio_base64": "string (required)",  // Base64-encoded audio data
  "sample_rate": 16000,                  // Optional, default: 16000 Hz
  "history": [],                         // Optional, conversation history
  "source": null,                        // Optional, document source filter
  "k": 5,                                // Optional, RAG retrieval count (1-20)
  "temperature": 0.2,                    // Optional, LLM temperature (0.0-2.0)
  "whisper_model": null,                 // Optional, Whisper model size
  "tts_voice": null,                     // Optional, TTS voice
  "tts_rate": null,                      // Optional, TTS rate
  "tts_pitch": null                      // Optional, TTS pitch
}
```

## Response Structure

```json
{
  "status": "success|partial|tts_failed",
  "transcription": "User's spoken text",
  "answer": "Assistant's response",
  "audio_base64": "Base64-encoded MP3 response (or null if tts_failed)",
  "timings": {
    "audio_decode_s": 0.1234,
    "stt_s": 1.5678,
    "rag_s": 0.8901,
    "tts_s": 0.2345,
    "total_s": 2.8158
  },
  "model_used": null,
  "provider_used": null
}
```

## Installation

Install or update dependencies:

```bash
cd voice
pip install -r requirements.txt
```

If FFmpeg is not installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

## Troubleshooting

### "Invalid Base64 encoding"
- Make sure you're encoding the entire audio file correctly
- Use `base64 -w 0 audio.wav` to ensure no newlines are added
- Check that the file is not empty

### "Decoded audio data is empty"
- Your audio file might be corrupted
- Try a different audio file format
- Make sure the file is at least a few hundred bytes

### "Failed to decode audio" (after fixes)
- FFmpeg might not be installed
- Audio format might not be supported
- Try converting to WAV first: `ffmpeg -i input.mp3 -acodec pcm_s16le -ar 16000 output.wav`

### "No speech detected"
- Audio volume too low (RMS < 0.01)
- Audio contains only silence
- Try increasing microphone volume or using a different recording

## Example Workflow

```bash
# 1. Record audio (uses sounddevice)
python -c "
import sounddevice as sd
import soundfile as sf
import numpy as np

print('Recording 5 seconds...')
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
sf.write('my_question.wav', audio, 16000)
print('Saved to my_question.wav')
"

# 2. Test the voice chat endpoint
python test_voice_client.py --audio my_question.wav

# 3. Check response
# You should see transcription and answer
```

## Performance Notes

- Audio decode: Usually < 1s (depends on file size)
- STT (Whisper): 1-5s (depends on Whisper model size)
- RAG: 0.5-5s (depends on backend load)
- TTS (Text-to-Speech): 0.2-2s
- **Total time: 2-15 seconds** (normal for first request as models load)

Use smaller Whisper models for faster processing:
- `tiny`: Fastest, ~100MB
- `base`: Balanced, ~140MB (default)
- `small`: Better, ~500MB
- `medium`: Best, ~1.5GB
