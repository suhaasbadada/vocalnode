import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
import inspect

print(dir(ChatterboxTurboTTS))
model = ChatterboxTurboTTS.from_pretrained(device="cpu")
print("generate spec:", inspect.signature(model.generate))
if hasattr(model, 'generate_stream'):
    print("generate_stream spec:", inspect.signature(model.generate_stream))
