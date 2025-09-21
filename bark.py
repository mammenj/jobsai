from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
text1 = "Hi my name is John [pause 0.5] I am a software developer. [pause 0.3] I love coding in Python.[laugh]"
text = ""
with open("script.txt", "r") as file:
    line = file.readline()
    while line:
        text += line
        line = file.readline()

tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)

# cloning `lj` voice from `TTS/tts/utils/assets/tortoise/voices/lj`
# with custom inference settings overriding defaults.
# tts.tts_to_file(
#    text=text,
#    file_path="output.wav",
#    voice_dir="voicedir/",
#    speaker="lj",
#    num_autoregressive_samples=1,
#    diffusion_iterations=10,
# )

# Using presets with the same voice
# tts.tts_to_file(
#    text=text1,
#    file_path="out/bark_output.wav",
#    # voice_dir="voicedir/",
#    speaker="jsmammen",
#    speaker_wav=["voicedir/new_speaker/myvoice.wav"],
# )

tts.tts_to_file(text=text, file_path="out/bark_output3.wav", speaker="jsmammen")


# Random voice generation
# tts.tts_to_file(text=text,
#                file_path="output.wav")
