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
