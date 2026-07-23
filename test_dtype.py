import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device="mps")
print(model.dtype if hasattr(model, "dtype") else "no dtype")
print(next(model.parameters()).dtype)
