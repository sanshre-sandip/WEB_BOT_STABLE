#!/usr/bin/env python3
"""Test client for /voice/chat endpoint - shows how to properly encode and send audio."""

import base64
import json
import sys
from pathlib import Path

import httpx
import numpy as np

def encode_audio_to_base64(audio_file_path: str | Path) -> str:
    """Read an audio file and encode it as Base64.
    
    Supports WAV, MP3, WebM, and other formats that FFmpeg can handle.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Base64-encoded audio string
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        ValueError: If the file is empty
    """
    audio_path = Path(audio_file_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read the binary audio data
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    if not audio_bytes:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    # Encode as Base64
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    print(f"✓ Encoded audio file: {audio_path}")
    print(f"  File size: {len(audio_bytes)} bytes")
    print(f"  Base64 length: {len(audio_base64)} chars")
    
    return audio_base64


def generate_test_audio(output_path: str | Path, duration_seconds: float = 2.0, sample_rate: int = 16000) -> Path:
    """Generate a simple sine wave test audio file (WAV format).
    
    Args:
        output_path: Where to save the test audio file
        duration_seconds: Length of the audio
        sample_rate: Sample rate in Hz
        
    Returns:
        Path to the generated audio file
    """
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Install with: pip install soundfile")
        sys.exit(1)
    
    output_path = Path(output_path)
    
    # Generate a simple sine wave at 440 Hz (A4 note)
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
    frequency = 440  # Hz
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Save as WAV
    sf.write(output_path, audio_data, sample_rate)
    
    print(f"✓ Generated test audio: {output_path}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Frequency: {frequency} Hz")
    
    return output_path


async def send_voice_chat_request(
    audio_base64: str,
    backend_url: str = "http://localhost:8011",
    history: list[dict[str, str]] | None = None,
    sample_rate: int = 16000,
) -> dict:
    """Send audio to the /voice/chat endpoint.
    
    Args:
        audio_base64: Base64-encoded audio data
        backend_url: URL of the backend server
        history: Conversation history (optional)
        sample_rate: Sample rate of the audio in Hz
        
    Returns:
        Response from the server
        
    Raises:
        httpx.HTTPError: If the request fails
    """
    backend_url = backend_url.rstrip("/")
    
    payload = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
        "history": history or [],
        "k": 5,
        "temperature": 0.2,
    }
    
    print(f"\n📤 Sending request to {backend_url}/voice/chat")
    print(f"   Payload size: {len(json.dumps(payload)) / 1024 / 1024:.2f} MB")
    
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                f"{backend_url}/voice/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            print(f"❌ Request failed with status {exc.response.status_code}")
            print(f"   Error: {exc.response.text}")
            raise
        except Exception as exc:
            print(f"❌ Request failed: {exc}")
            raise


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test /voice/chat endpoint")
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file (WAV, MP3, WebM, etc.)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a test audio file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="http://localhost:8011",
        help="Backend URL (default: http://localhost:8011)",
    )
    parser.add_argument(
        "--history",
        type=str,
        help="Conversation history as JSON string",
    )
    
    args = parser.parse_args()
    
    # Get or generate audio file
    if args.generate:
        audio_path = generate_test_audio("test_audio.wav")
    elif args.audio:
        audio_path = args.audio
    else:
        print("ERROR: Provide --audio <path> or use --generate")
        parser.print_help()
        sys.exit(1)
    
    # Encode audio to Base64
    try:
        audio_base64 = encode_audio_to_base64(audio_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    
    # Parse history if provided
    history = []
    if args.history:
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON history: {exc}")
            sys.exit(1)
    
    # Send request
    try:
        response = await send_voice_chat_request(
            audio_base64,
            backend_url=args.backend,
            history=history,
        )
        
        print("\n✅ Success!")
        print(f"   Transcription: {response.get('transcription', 'N/A')}")
        print(f"   Answer: {response.get('answer', 'N/A')}")
        print(f"   Status: {response.get('status', 'N/A')}")
        print(f"\n⏱️  Timings:")
        for key, value in response.get('timings', {}).items():
            print(f"   {key}: {value}s")
        
        if response.get('audio_base64'):
            print(f"   Audio response size: {len(response['audio_base64']) / 1024:.1f} KB")
        
    except Exception as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
