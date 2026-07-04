import wave
from unittest.mock import patch

import numpy as np

from mcp_server.audio_hardware_mcp import (
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    _play_audio,
    _record_audio,
)


def _write_wav(path, channels, sample_width, framerate, frames_bytes):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames_bytes)


def test_record_audio_writes_valid_mono_wav_file(tmp_path):
    duration = 0.5
    expected_frames = int(duration * SAMPLE_RATE)
    fake_audio = np.zeros((expected_frames, CHANNELS), dtype="int16")
    output_path = str(tmp_path / "recording.wav")

    with patch("mcp_server.audio_hardware_mcp.sd.rec", return_value=fake_audio) as mock_rec, \
         patch("mcp_server.audio_hardware_mcp.sd.wait") as mock_wait:
        result = _record_audio(duration, output_path)

    mock_rec.assert_called_once_with(
        expected_frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
    )
    mock_wait.assert_called_once()
    assert result == output_path

    with wave.open(output_path, "rb") as wav_file:
        assert wav_file.getnchannels() == CHANNELS
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getsampwidth() == SAMPLE_WIDTH
        assert wav_file.getnframes() == expected_frames


def test_record_audio_rounds_duration_to_nearest_frame_count(tmp_path):
    duration = 0.33
    expected_frames = int(duration * SAMPLE_RATE)
    fake_audio = np.zeros((expected_frames, CHANNELS), dtype="int16")
    output_path = str(tmp_path / "short.wav")

    with patch("mcp_server.audio_hardware_mcp.sd.rec", return_value=fake_audio) as mock_rec, \
         patch("mcp_server.audio_hardware_mcp.sd.wait"):
        _record_audio(duration, output_path)

    mock_rec.assert_called_once_with(
        expected_frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16"
    )


def test_play_audio_plays_mono_wav_with_correct_samplerate(tmp_path):
    samples = np.array([0, 100, -100, 200], dtype="int16")
    input_path = tmp_path / "mono.wav"
    _write_wav(input_path, channels=1, sample_width=2, framerate=22050, frames_bytes=samples.tobytes())

    with patch("mcp_server.audio_hardware_mcp.sd.play") as mock_play, \
         patch("mcp_server.audio_hardware_mcp.sd.wait") as mock_wait:
        result = _play_audio(str(input_path))

    mock_wait.assert_called_once()
    assert mock_play.call_count == 1
    played_data = mock_play.call_args[0][0]
    played_kwargs = mock_play.call_args[1]
    assert played_kwargs["samplerate"] == 22050
    np.testing.assert_array_equal(played_data, samples)
    assert result == f"Played {input_path}"


def test_play_audio_reshapes_stereo_wav(tmp_path):
    samples = np.array([0, 1, 2, 3, 4, 5], dtype="int16")
    input_path = tmp_path / "stereo.wav"
    _write_wav(input_path, channels=2, sample_width=2, framerate=16000, frames_bytes=samples.tobytes())

    with patch("mcp_server.audio_hardware_mcp.sd.play") as mock_play, \
         patch("mcp_server.audio_hardware_mcp.sd.wait"):
        _play_audio(str(input_path))

    played_data = mock_play.call_args[0][0]
    assert played_data.shape == (3, 2)


def test_record_and_play_tools_are_registered_on_mcp():
    import asyncio

    from mcp_server.audio_hardware_mcp import mcp

    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert "record_audio" in tool_names
    assert "play_audio" in tool_names
