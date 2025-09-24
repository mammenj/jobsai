import torch
from TTS.api import TTS
import time

start = time.perf_counter()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


# Initialize a TTS model with voice cloning support
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 1. Clone the voice from `speaker_wav` and cache it under a custom speaker ID
tts.tts_to_file(
    text="Hello world",
    speaker_wav=["voicedir/jsmammen.wav", "voicedir/jsmammen2.wav"],
    speaker="newjsmammen",
    language="en",
)

# 2. The voice can now be reused without providing reference audio
tts.tts_to_file(
    text="Hello world",
    speaker="newjsmammen",
    language="en",
)


end = time.perf_counter()
print(f"Execution time: {end - start:0.4f} seconds")
