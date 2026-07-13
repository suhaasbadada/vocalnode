import wave

import numpy as np
import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


class _FakeVoiceModel:
    def __init__(self, sr=24000, generated_samples=None):
        self.prepare_conditionals_calls = []
        self.generate_calls = []
        self.sr = sr
        self._generated_samples = (
            generated_samples if generated_samples is not None else np.zeros(100, dtype="float32")
        )

    def prepare_conditionals(self, reference_path):
        self.prepare_conditionals_calls.append(reference_path)

    def generate(self, text):
        self.generate_calls.append(text)
        return torch.from_numpy(self._generated_samples).unsqueeze(0)


def _patch_default_voice(monkeypatch, model=None):
    """Reset the module-level voice-model singleton and stub out the real
    ChatterboxTTS loader so tests never need the actual (uninstalled)
    chatterbox package.
    """
    fake_model = model if model is not None else _FakeVoiceModel()
    monkeypatch.setattr(main_module, "_tts_model", None)
    monkeypatch.setattr(main_module, "_default_voice_ready", False)
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
    assert str(missing_path) in response.json()["detail"]


def test_trigger_wakeword_rejects_missing_field():
    response = client.post("/trigger-wakeword", json={})

    assert response.status_code == 422


def test_trigger_wakeword_rejects_path_that_is_a_directory(tmp_path):
    response = client.post("/trigger-wakeword", json={"audio_file_path": str(tmp_path)})

    assert response.status_code == 404


def test_generate_speech_returns_success_and_writes_wav_file(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    _patch_default_voice(monkeypatch, model=_FakeVoiceModel(sr=24000, generated_samples=np.linspace(-1, 1, 4800, dtype="float32")))

    response = client.post("/generate-speech", json={"text": "Hello there"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["output_path"] == str(output_path)
    assert output_path.exists()

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000  # matches the fake model's sr, not a hardcoded value
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 4800


def test_generate_speech_rejects_missing_text_field():
    response = client.post("/generate-speech", json={})

    assert response.status_code == 422


def test_generate_speech_calls_model_generate_with_request_text(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    fake_model = _patch_default_voice(monkeypatch)

    response = client.post("/generate-speech", json={"text": "synthesize me"})

    assert response.status_code == 200
    assert fake_model.generate_calls == ["synthesize me"]


def test_generate_speech_uses_models_own_sample_rate_not_a_hardcoded_one(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    _patch_default_voice(monkeypatch, model=_FakeVoiceModel(sr=16000, generated_samples=np.zeros(1600, dtype="float32")))

    client.post("/generate-speech", json={"text": "hi"})

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000


def test_generate_speech_clips_out_of_range_samples_to_valid_int16(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    out_of_range_samples = np.array([-2.0, 2.0, 0.0], dtype="float32")
    _patch_default_voice(monkeypatch, model=_FakeVoiceModel(sr=24000, generated_samples=out_of_range_samples))

    response = client.post("/generate-speech", json={"text": "hi"})

    assert response.status_code == 200
    with wave.open(str(output_path), "rb") as wav_file:
        raw = wav_file.readframes(wav_file.getnframes())
        decoded = np.frombuffer(raw, dtype="int16")
        assert decoded[0] == -32767
        assert decoded[1] == 32767
        assert decoded[2] == 0


def test_generate_speech_extracts_default_voice_embedding_from_clone_reference(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    fake_model = _patch_default_voice(monkeypatch)

    client.post("/generate-speech", json={"text": "Hello there"})

    assert fake_model.prepare_conditionals_calls == [str(main_module.VOICE_CLONE_REFERENCE_PATH)]


def test_generate_speech_only_extracts_default_voice_embedding_once(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)
    fake_model = _patch_default_voice(monkeypatch)

    client.post("/generate-speech", json={"text": "first request"})
    client.post("/generate-speech", json={"text": "second request"})

    assert len(fake_model.prepare_conditionals_calls) == 1


def test_generate_speech_reuses_cached_voice_model_across_requests(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)

    load_calls = []
    fake_model = _FakeVoiceModel()

    def fake_loader():
        load_calls.append(1)
        return fake_model

    monkeypatch.setattr(main_module, "_tts_model", None)
    monkeypatch.setattr(main_module, "_default_voice_ready", False)
    monkeypatch.setattr(main_module, "_load_tts_model", fake_loader)

    client.post("/generate-speech", json={"text": "first request"})
    client.post("/generate-speech", json={"text": "second request"})

    assert len(fake_model.prepare_conditionals_calls) == 1
    assert len(load_calls) == 2  # cheap idempotent check each time; only extracts once
