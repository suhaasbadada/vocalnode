import asyncio
import wave
from pathlib import Path

import numpy as np

import agent_loop


def _write_test_wav(path: Path, sample_rate: int = 16000, num_samples: int = 16000) -> None:
    samples = np.zeros(num_samples, dtype="int16")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


class _DummyMcpClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _patch_client(monkeypatch):
    monkeypatch.setattr(agent_loop, "Client", lambda server: _DummyMcpClient())


def _patch_helpers(monkeypatch, calls, wake_detected):
    async def fake_record(client, duration, output_path):
        calls["record_durations"].append(duration)
        return output_path

    async def fake_play(client, file_path):
        calls["play_paths"].append(file_path)
        return f"played {file_path}"

    async def fake_speech(text, url=agent_loop.GENERATE_SPEECH_URL):
        calls["speech_texts"].append(text)
        return {"status": "success"}

    def fake_detect(interpreter, wav_path, threshold=agent_loop.DETECTION_THRESHOLD):
        calls["detect_calls"].append(wav_path)
        return wake_detected

    monkeypatch.setattr(agent_loop, "record_via_mcp", fake_record)
    monkeypatch.setattr(agent_loop, "play_via_mcp", fake_play)
    monkeypatch.setattr(agent_loop, "request_generated_speech", fake_speech)
    monkeypatch.setattr(agent_loop, "detect_wake_word", fake_detect)


def test_run_agent_loop_skips_response_when_wake_word_not_detected(monkeypatch):
    calls = {"record_durations": [], "play_paths": [], "speech_texts": [], "detect_calls": []}
    _patch_client(monkeypatch)
    _patch_helpers(monkeypatch, calls, wake_detected=False)

    asyncio.run(agent_loop.run_agent_loop(interpreter=object(), max_iterations=1))

    assert calls["record_durations"] == [agent_loop.LISTEN_CHUNK_DURATION_S]
    assert calls["speech_texts"] == []
    assert calls["play_paths"] == []


def test_run_agent_loop_responds_when_wake_word_detected(monkeypatch):
    calls = {"record_durations": [], "play_paths": [], "speech_texts": [], "detect_calls": []}
    _patch_client(monkeypatch)
    _patch_helpers(monkeypatch, calls, wake_detected=True)

    asyncio.run(agent_loop.run_agent_loop(interpreter=object(), max_iterations=1))

    assert calls["record_durations"] == [
        agent_loop.LISTEN_CHUNK_DURATION_S,
        agent_loop.COMMAND_CAPTURE_DURATION_S,
    ]
    assert calls["speech_texts"] == [agent_loop.HARDCODED_RESPONSE_TEXT]
    assert calls["play_paths"] == [agent_loop.GENERATED_SPEECH_PATH]


def test_run_agent_loop_runs_multiple_continuous_cycles(monkeypatch):
    calls = {"record_durations": [], "play_paths": [], "speech_texts": [], "detect_calls": []}
    _patch_client(monkeypatch)
    _patch_helpers(monkeypatch, calls, wake_detected=False)

    asyncio.run(agent_loop.run_agent_loop(interpreter=object(), max_iterations=3))

    assert calls["record_durations"] == [agent_loop.LISTEN_CHUNK_DURATION_S] * 3


def test_run_agent_loop_stops_immediately_when_stop_event_already_set(monkeypatch):
    calls = {"record_durations": [], "play_paths": [], "speech_texts": [], "detect_calls": []}
    _patch_client(monkeypatch)
    _patch_helpers(monkeypatch, calls, wake_detected=False)

    async def _run():
        stop_event = asyncio.Event()
        stop_event.set()
        await agent_loop.run_agent_loop(interpreter=object(), stop_event=stop_event)

    asyncio.run(_run())

    assert calls["record_durations"] == []


def test_extract_features_returns_expected_shape_and_dtype(tmp_path):
    wav_path = tmp_path / "chunk.wav"
    _write_test_wav(wav_path, sample_rate=agent_loop.SAMPLE_RATE_HZ, num_samples=agent_loop.SAMPLE_RATE_HZ * 2)

    features = agent_loop._extract_features(wav_path)

    assert features.shape == (agent_loop.SPECTROGRAM_LENGTH, agent_loop.FEATURE_COUNT)
    assert features.dtype == np.float32


def test_detect_wake_word_uses_threshold_against_model_score(tmp_path, monkeypatch):
    wav_path = tmp_path / "chunk.wav"
    _write_test_wav(wav_path, sample_rate=agent_loop.SAMPLE_RATE_HZ, num_samples=agent_loop.SAMPLE_RATE_HZ * 2)

    monkeypatch.setattr(
        agent_loop,
        "_extract_features",
        lambda path: np.zeros((agent_loop.SPECTROGRAM_LENGTH, agent_loop.FEATURE_COUNT), dtype="float32"),
    )

    class DummyInterpreter:
        def get_input_details(self):
            return [{"index": 0, "dtype": np.float32}]

        def get_output_details(self):
            return [{"index": 0}]

        def set_tensor(self, index, value):
            self._value = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.array([[0.95]], dtype="float32")

    assert agent_loop.detect_wake_word(DummyInterpreter(), wav_path, threshold=0.9) is True
    assert agent_loop.detect_wake_word(DummyInterpreter(), wav_path, threshold=0.99) is False
