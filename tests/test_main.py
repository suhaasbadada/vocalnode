import wave
import io

import numpy as np
import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)

class _FakeVoiceModel:
    def __init__(self, sr=24000, generated_samples=None):
        self.generate_calls = []
        self.sr = sr
        self._generated_samples = (
            generated_samples if generated_samples is not None else np.zeros(100, dtype="float32")
        )

    def generate(self, text, audio_prompt_path=None):
        self.generate_calls.append((text, audio_prompt_path))
        return torch.from_numpy(self._generated_samples).unsqueeze(0)

def _patch_default_voice(monkeypatch, model=None):
    fake_model = model if model is not None else _FakeVoiceModel()
    monkeypatch.setattr(main_module, "_tts_model", None)
    monkeypatch.setattr(main_module, "_load_tts_model", lambda: fake_model)
    return fake_model

def test_trigger_wakeword_returns_success_for_existing_file(tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"placeholder-audio-bytes")

    response = client.post("/trigger-wakeword", json={"audio_file_path": str(audio_file)})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert str(audio_file) in body["message"]

def test_trigger_wakeword_returns_404_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.wav"
    response = client.post("/trigger-wakeword", json={"audio_file_path": str(missing_path)})
    assert response.status_code == 404

def test_upload_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "OUTPUT_AUDIO_DIR", tmp_path)
    file_content = b"fake audio content"
    
    response = client.post(
        "/voice", 
        files={"audio": ("test_voice.wav", file_content, "audio/wav")}
    )
    
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "voice_id" in body
    
    voice_id = body["voice_id"]
    saved_file = tmp_path / f"{voice_id}.wav"
    assert saved_file.exists()
    assert saved_file.read_bytes() == file_content

def test_chunk_text_preserves_brackets():
    text = "Hello there! [laugh] How are you?"
    chunks = main_module.chunk_text(text)
    assert chunks == ["Hello there!", "[laugh] How are you?"]

def test_generate_speech_streams_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "OUTPUT_AUDIO_DIR", tmp_path)
    fake_model = _patch_default_voice(monkeypatch)

    response = client.post("/generate-speech", json={"text": "Hello, world!"})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
        
    assert len(content) > 0
    assert fake_model.generate_calls == [("Hello,", None), ("world!", None)]

def test_generate_speech_with_voice_id(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "OUTPUT_AUDIO_DIR", tmp_path)
    fake_model = _patch_default_voice(monkeypatch)
    
    # Create fake voice file
    voice_id = "test-123"
    voice_file = tmp_path / f"{voice_id}.wav"
    voice_file.write_bytes(b"dummy")

    response = client.post("/generate-speech", json={"text": "Hi.", "voice_id": voice_id})
    
    assert response.status_code == 200
    list(response.iter_bytes()) # consume stream
    
    assert fake_model.generate_calls == [("Hi.", str(voice_file))]
