# vocalnode

A local voice assistant that listens for a custom wake word ("Hey Chatter"),
captures your command, and replies with synthesized, voice-cloned speech.

Built from two open-source models glued together with a FastAPI backend:

- **[microWakeWord](https://github.com/kahrendt/microWakeWord)** — detects
  the "Hey Chatter" wake word from a quantized `.tflite` model.
- **[ChatterboxTTS](https://github.com/resemble-ai/chatterbox)** — synthesizes
  the spoken reply, cloned to sound like a reference voice you provide.

## How it works

```
mic input --> [microWakeWord] --> wake word detected?
                                       |
                                       v
                        record command --> POST /generate-speech
                                       |
                                       v
                        ChatterboxTTS synthesizes reply
                        (cloned to your reference voice)
                                       |
                                       v
                              played back through speakers
```

- `app/main.py` — FastAPI server exposing `/generate-speech` (ChatterboxTTS
  synthesis) and `/trigger-wakeword`.
- `agent_loop.py` — the always-on daemon: records audio, runs wake word
  detection, and calls the FastAPI server to synthesize + play a reply.
- `mcp_server/audio_hardware_mcp.py` — MCP server exposing `record_audio` /
  `play_audio` tools over the machine's default input/output devices.
- `train_wakeword.py` — trains the "Hey Chatter" wake word model and exports
  it to `models/hey_chatter.tflite`.

## Requirements

Install Python dependencies with `pip`, plus the system packages listed in
[`system_reqs.txt`](system_reqs.txt) (`ffmpeg`, `portaudio`, a pinned
`setuptools` version — see that file for why each is needed and how to
install it for your OS).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r system_reqs.txt   # installs Python packages; see file for OS-level deps too
```

## Setup

1. **Provide a voice to clone.** Drop a short (~10–15s), clean,
   single-speaker reference clip with no music/silence/other voices at
   `audio_data/KR_clone.wav`. This is the voice every synthesized reply will
   be cloned to. See
   [`.claude/skills/FormatAudioForTTS.md`](.claude/skills/FormatAudioForTTS.md)
   for how to convert/trim a raw recording into a usable reference clip.

2. **Train the wake word model** (only needed once, or whenever you want to
   retrain):
   ```bash
   python train_wakeword.py
   ```
   This produces `models/hey_chatter.tflite`.

## Running

Start the FastAPI server (handles speech synthesis):

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a separate terminal, start the always-on wake word daemon:

```bash
source .venv/bin/activate
python agent_loop.py
```

Say "Hey Chatter" — the daemon will record your command, ask the FastAPI
server to synthesize a reply in the cloned voice, and play it back.

## Testing

### Automated tests

```bash
pytest
```

### Manually testing speech synthesis

With the FastAPI server running, hit the endpoint directly:

```bash
curl -X POST http://localhost:8000/generate-speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test."}'
```

This writes the result to `audio_data/output_response.wav`. Play it back and
compare it against your reference clip to judge cloning quality:

```bash
afplay audio_data/output_response.wav   # the synthesized reply
afplay audio_data/KR_clone.wav          # the reference voice it was cloned from
```

## Notes

- `audio_data/`, `*.wav`, `*.pt`, `*.tflite`, and `*.onnx` are gitignored —
  audio and model artifacts are local-only, not version controlled.
- The first `/generate-speech` call in a fresh server process is slow (model
  download/load + reference voice conditioning); every call after that only
  pays the synthesis cost, since the model and voice embedding are cached in
  memory for the life of the process.
