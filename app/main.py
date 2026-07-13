import asyncio
import subprocess
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

OUTPUT_AUDIO_DIR = Path("audio_data")
OUTPUT_AUDIO_PATH = OUTPUT_AUDIO_DIR / "output_response.wav"
VOICE_CLONE_REFERENCE_PATH = OUTPUT_AUDIO_DIR / "KR2_mvsep_clone.wav"

# The 22050Hz spec (see .claude/skills/FormatAudioForTTS.md) is for reference
# / voice-clone *input* audio. ChatterboxTTS's synthesized *output* comes out
# at its own native rate (model.sr, 24000Hz for the current model) -- the WAV
# file we write must use that actual rate, not a hardcoded one.
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # bytes per sample (int16)

# Lazily-loaded ChatterboxTTS singleton and default-voice-loaded flag. Kept
# as module globals so the (heavy) model is loaded once and its extracted
# speaker embedding is reused as the default voice for every subsequent
# /generate-speech call, rather than reloading/re-extracting per request.
_tts_model = None
_default_voice_ready = False


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


def _load_tts_model():
    """Lazily load and cache the ChatterboxTurboTTS model.

    The import is deferred so this module stays importable (and testable)
    without the chatterbox-tts package installed; it's only pulled in once
    the endpoint is actually used. Turbo trades some of the exaggeration/cfg
    controls for speed (fewer vocoder steps, faster token decoding); device
    picks the fastest backend available (CUDA > Apple MPS > CPU).
    """
    global _tts_model
    if _tts_model is None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _tts_model = ChatterboxTurboTTS.from_pretrained(device=device)
    return _tts_model


def _ensure_default_voice_loaded(reference_wav_path: Path = VOICE_CLONE_REFERENCE_PATH) -> None:
    """Load the reference clone audio and extract its speaker embedding via
    ChatterboxTTS's conditioning pipeline, caching it as the default voice.

    prepare_conditionals() extracts the speaker embedding from the reference
    clip and stores it on the model, so every subsequent model.generate(text)
    call reuses it automatically without needing the reference audio again.
    This only runs once per process -- later calls are a no-op check.
    """
    global _default_voice_ready
    if _default_voice_ready:
        return

    model = _load_tts_model()
    model.prepare_conditionals(str(reference_wav_path))
    _default_voice_ready = True


def _generate_speech_audio(text: str) -> tuple[bytes, int]:
    """Synthesize `text` with ChatterboxTTS using the cached default voice
    (set up by _ensure_default_voice_loaded), returning 16-bit PCM bytes and
    the sample rate they were generated at.
    """
    model = _load_tts_model()
    wav_tensor = model.generate(text)  # shape (1, num_samples), float32 in [-1, 1]

    samples = wav_tensor.squeeze(0).detach().cpu().numpy()
    clipped = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).astype("int16")

    return pcm_int16.tobytes(), model.sr


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(TTS_CHANNELS)
        wav_file.setsampwidth(TTS_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return path


@app.post("/generate-speech")
async def generate_speech(request: GenerateSpeechRequest):
    # Model loading, speech synthesis, and the file write are all pushed
    # onto a worker thread so the event loop stays free to handle other
    # requests concurrently instead of blocking on CPU/disk work.
    await asyncio.to_thread(_ensure_default_voice_loaded)
    pcm_bytes, sample_rate = await asyncio.to_thread(_generate_speech_audio, request.text)
    output_path = await asyncio.to_thread(_write_wav, OUTPUT_AUDIO_PATH, pcm_bytes, sample_rate)

    return {
        "status": "success",
        "message": f"Generated speech for text of length {len(request.text)}",
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
