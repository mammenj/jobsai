import torch
from TTS.api import TTS

# text = "    We talked of the side show in the circus.    Use a pencil to write the first draft.    He ran half way to the hardware store.    The clock struck to mark the third period.    A small creek cutacross the field.    Cars and busses stalled in snow drifts.    The set of china hit the floor with a crash.    This is a grand season for hikes on the road.    The dune rose from the edge of the water.   Those words were the cue for the actor to leave."


text = ""
with open("script.txt", "r") as file:
    line = file.readline()
    while line:
        text += line
        line = file.readline()




# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# Text to speech to a file
tts.tts_to_file(text=text, speaker_wav="voicedir/myvoice.wav", language="en", file_path="out/xtts2_output.wav")
