import wave

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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

    response = client.post("/generate-speech", json={"text": "Hello there"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["output_path"] == str(output_path)
    assert output_path.exists()

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 22050
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() > 0


def test_generate_speech_rejects_missing_text_field():
    response = client.post("/generate-speech", json={})

    assert response.status_code == 422


def test_generate_speech_calls_placeholder_tts_with_request_text(tmp_path, monkeypatch):
    output_path = tmp_path / "output_response.wav"
    monkeypatch.setattr("app.main.OUTPUT_AUDIO_PATH", output_path)

    captured = {}

    def fake_tts(text):
        captured["text"] = text
        return b"\x00\x00"

    monkeypatch.setattr("app.main._chatterbox_tts_placeholder", fake_tts)

    response = client.post("/generate-speech", json={"text": "synthesize me"})

    assert response.status_code == 200
    assert captured["text"] == "synthesize me"
