import wave

import numpy as np
import sounddevice as sd
from fastmcp import FastMCP

mcp = FastMCP("audio-hardware")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)

_DTYPE_BY_SAMPLE_WIDTH = {1: "int8", 2: "int16", 4: "int32"}


def _record_audio(duration: float, output_path: str) -> str:
    frame_count = int(duration * SAMPLE_RATE)
    frames = sd.rec(frame_count, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(frames.tobytes())

    return output_path


def _play_audio(file_path: str) -> str:
    with wave.open(file_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw_frames = wav_file.readframes(wav_file.getnframes())

    dtype = _DTYPE_BY_SAMPLE_WIDTH.get(sample_width, "int16")
    data = np.frombuffer(raw_frames, dtype=dtype)
    if channels > 1:
        data = data.reshape(-1, channels)

    sd.play(data, samplerate=sample_rate)
    sd.wait()

    return f"Played {file_path}"


@mcp.tool
def record_audio(duration: float, output_path: str) -> str:
    """Record mono audio from the default input device and save it as a WAV file.

    Args:
        duration: Recording length in seconds.
        output_path: Destination path for the recorded WAV file.

    Returns:
        The output_path the recording was saved to.
    """
    return _record_audio(duration, output_path)


@mcp.tool
def play_audio(file_path: str) -> str:
    """Play back a WAV file through the default output device.

    Args:
        file_path: Path to the WAV file to play.

    Returns:
        A confirmation message once playback completes.
    """
    return _play_audio(file_path)


if __name__ == "__main__":
    mcp.run()
