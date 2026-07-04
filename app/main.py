import asyncio
import subprocess
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

OUTPUT_AUDIO_DIR = Path("audio_data")
OUTPUT_AUDIO_PATH = OUTPUT_AUDIO_DIR / "output_response.wav"

TTS_SAMPLE_RATE = 22050  # ChatterboxTTS output spec
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # bytes per sample (int16)


class TriggerWakewordRequest(BaseModel):
    audio_file_path: str


class GenerateSpeechRequest(BaseModel):
    text: str


def _file_exists_locally(path: str) -> bool:
    result = subprocess.run(["test", "-f", path], capture_output=True)
    return result.returncode == 0


@app.post("/trigger-wakeword")
def trigger_wakeword(request: TriggerWakewordRequest):
    if not _file_exists_locally(request.audio_file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {request.audio_file_path}",
        )

    # TTS/wakeword processing logic not implemented yet.
    return {
        "status": "success",
        "message": f"Wakeword trigger received for {request.audio_file_path}",
    }


def _chatterbox_tts_placeholder(text: str) -> bytes:
    """Placeholder for ChatterboxTTS speech generation.

    Real synthesis is not implemented yet; this returns silent PCM data
    proportional to the input text length so the request -> synthesis ->
    WAV file pipeline can be exercised end-to-end.
    """
    frame_count = max(TTS_SAMPLE_RATE // 2, len(text) * 200)
    return b"\x00" * (frame_count * TTS_SAMPLE_WIDTH)


def _write_wav(path: Path, pcm_bytes: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(TTS_CHANNELS)
        wav_file.setsampwidth(TTS_SAMPLE_WIDTH)
        wav_file.setframerate(TTS_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return path


@app.post("/generate-speech")
async def generate_speech(request: GenerateSpeechRequest):
    # Both the placeholder synthesis step and the file write are pushed onto
    # a worker thread so the event loop stays free to handle other requests
    # concurrently instead of blocking on CPU/disk work.
    pcm_bytes = await asyncio.to_thread(_chatterbox_tts_placeholder, request.text)
    output_path = await asyncio.to_thread(_write_wav, OUTPUT_AUDIO_PATH, pcm_bytes)

    return {
        "status": "success",
        "message": f"Generated speech for text of length {len(request.text)}",
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
