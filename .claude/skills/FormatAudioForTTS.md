# Formatting Audio for ChatterboxTTS

ChatterboxTTS expects mono, 22050Hz input audio. Any file that doesn't already
match that spec must be converted with `ffmpeg` before it's used for training,
cloning, or inference reference audio.

## Steps

1. Confirm the input file exists and identify its current channel count and
   sample rate:
   ```bash
   ffprobe -v error -show_entries stream=channels,sample_rate -of default=noprint_wrappers=1 <input_file>
   ```
2. Downmix to a single mono channel and resample to 22050Hz using `ffmpeg`:
   ```bash
   ffmpeg -i <input_file> -ac 1 -ar 22050 <output_file>
   ```
   - `-ac 1` forces mono downmix.
   - `-ar 22050` forces the 22050Hz sample rate ChatterboxTTS requires.
3. Verify the output matches spec by re-running `ffprobe` on `<output_file>`
   and confirming `channels=1` and `sample_rate=22050`.
4. Only pass the converted `<output_file>` into the TTS pipeline — never the
   original file if it didn't already match spec.

## Reminders

- `ffmpeg` itself is the required tool for this conversion — never reach for
  `apt-get`, `brew`, or any other system package manager to install codecs or
  audio tooling. The project's `block_audio_sys_changes.sh` PreToolUse hook
  will detect `apt-get` (and `brew`, `alsa`, `pulseaudio`) in a Bash command
  and exit 2, blocking the tool call outright.
- If `ffmpeg` is missing, it belongs in `system_reqs.txt` as a system
  dependency for the user to install themselves — do not attempt to install
  it directly.
- Never read the raw `.wav` file contents into context — only shell out to
  `ffmpeg`/`ffprobe` and inspect their text output.
