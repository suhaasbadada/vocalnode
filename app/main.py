import asyncio
import subprocess
import wave
import re
import uuid
import shutil
import json
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import pedalboard
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

OUTPUT_AUDIO_DIR = Path("audio_data")
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VOICES_REGISTRY_FILE = OUTPUT_AUDIO_DIR / "voices.json"

if not VOICES_REGISTRY_FILE.exists():
    VOICES_REGISTRY_FILE.write_text(json.dumps({}))

TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # bytes per sample (int16)

_tts_model = None

class TriggerWakewordRequest(BaseModel):
    audio_file_path: str

class GenerateSpeechRequest(BaseModel):
    text: str
    voice_id: str | None = None
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    exaggeration: float = 0.0
    speed: float = 1.0

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
    return {
        "status": "success",
        "message": f"Wakeword trigger received for {request.audio_file_path}",
    }

def _load_tts_model():
    """Lazily load and cache the ChatterboxTurboTTS model."""
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

@app.post("/voice")
async def upload_voice(audio: UploadFile = File(...), name: str = Form(None)):
    """Uploads an audio file to use as a voice clone reference/fingerprint."""
    voice_id = str(uuid.uuid4())
    # Try to extract a clean name from the filename or use custom name
    voice_name = name if name else (Path(audio.filename).stem if audio.filename else f"Voice {voice_id[:4]}")
    
    file_extension = Path(audio.filename).suffix if audio.filename else ".wav"
    output_path = OUTPUT_AUDIO_DIR / f"{voice_id}{file_extension}"
    
    content = await audio.read()
    with open(output_path, "wb") as buffer:
        buffer.write(content)
        
    # Update registry
    registry = json.loads(VOICES_REGISTRY_FILE.read_text())
    registry[voice_id] = {"name": voice_name, "id": voice_id}
    VOICES_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
        
    return {
        "status": "success",
        "voice_id": voice_id,
        "name": voice_name,
        "message": "Voice fingerprint saved successfully"
    }

@app.get("/voices")
def get_voices():
    """Returns a list of saved voice profiles."""
    registry = json.loads(VOICES_REGISTRY_FILE.read_text())
    return {"status": "success", "voices": list(registry.values())}

def chunk_text(text: str) -> list[str]:
    """Splits text intelligently by punctuation while preserving paralinguistic tags."""
    # Split by sentence-ending punctuation or commas, 
    # to keep chunks short for minimum TTFB latency.
    chunks = re.split(r'([.?!,])\s*', text)
    
    merged_chunks = []
    current_chunk = ""
    for part in chunks:
        current_chunk += part
        if part in ['.', '?', '!', ',']:
            if current_chunk.strip():
                merged_chunks.append(current_chunk.strip())
            current_chunk = " "
            
    if current_chunk.strip():
        merged_chunks.append(current_chunk.strip())
        
    return merged_chunks

def _generate_chunk_audio(chunk_text: str, voice_path: str | None, temperature: float, repetition_penalty: float, exaggeration: float, speed: float) -> bytes:
    """Generates audio bytes for a single text chunk."""
    model = _load_tts_model()
    # Pass audio_prompt_path directly to generate if a voice_path is provided
    # Note: If voice_path is None, it uses the base voice.
    wav_tensor = model.generate(
        chunk_text, 
        audio_prompt_path=voice_path,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        exaggeration=exaggeration
    )
    
    samples = wav_tensor.squeeze(0).detach().cpu().numpy()
    
    if speed != 1.0:
        # pedalboard.time_stretch works perfectly but returns shape (channels, samples).
        # We must squeeze it back to a 1D array so the fade-in/fade-out logic works!
        import pedalboard
        stretched = pedalboard.time_stretch(samples, 24000, speed)
        samples = stretched.squeeze()
    
    # Apply a 10ms fade-in and fade-out to prevent audio stitching clicks/pops
    fade_len = int(24000 * 0.01) # 10ms at 24kHz
    if len(samples) > fade_len * 2:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        samples[:fade_len] *= fade_in
        samples[-fade_len:] *= fade_out

    # Peak normalization to prevent hard clipping (which causes roughness/distortion)
    # AI TTS models often generate waveforms that exceed 1.0 amplitude
    peak = np.max(np.abs(samples))
    if peak > 0.95:
        samples = samples * (0.95 / peak)

    clipped = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).astype("int16")
    return pcm_int16.tobytes()

@app.post("/generate-speech")
async def generate_speech(request: GenerateSpeechRequest):
    """Streams audio generation phrase-by-phrase for instant playback."""
    voice_path = None
    if request.voice_id:
        # Find the file with this ID (ignoring extension)
        matches = list(OUTPUT_AUDIO_DIR.glob(f"{request.voice_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Voice fingerprint not found")
        voice_path = str(matches[0])

    chunks = chunk_text(request.text)

    async def audio_streamer():
        # First chunk gets generated and yielded immediately
        for chunk in chunks:
            if not chunk.strip():
                continue
            # Yield raw PCM bytes for each chunk
            pcm_bytes = await asyncio.to_thread(
                _generate_chunk_audio, 
                chunk, 
                voice_path,
                request.temperature,
                request.repetition_penalty,
                request.exaggeration,
                request.speed
            )
            yield pcm_bytes

    # Return a chunked streaming response
    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
