import torch
from TTS.api import TTS
import time

train = False
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Init TTS with the target model name
# tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(
#    device
# )

# Run TTS
# tts.tts_to_file(text=text, file_path="./test")

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/en/vctk/fast_pitch").to(device)

if train:
    tts.tts_to_file(
        text="This is voice cloning. I am john mammen [laughs] and I am testing yourtts model [sighs]",
        speaker_wav="voicedir/jsmammen.wav",
        speaker="jsmammen",
        language="en",
        file_path="out/your_tts/your_tts-output.wav",
    )
    print("TTS with voice cloning complete!")

count = 0
with open("script.txt", "r") as file:
    line = file.readline()
    while line:
        line = file.readline()
        print("Processing :: ", line)
        if line.strip() != "" and line.__len__() > 3:
            start = time.perf_counter()
            count += 1
            audio_path = f"out/fastpitch/fast_pitch_tts-output-{count:03d}.wav"
            tts.tts_to_file(
                line,
                speaker_id="VCTK_p225",
                speaker_wav="voicedir/jsmammen.wav",
                file_path=audio_path,
            )
            end = time.perf_counter()
            print(
                f"Time taken to process {count} :: {end - start:0.4f} seconds :: for {{audio_path}}"
            )

print("TTS with voice cloning completed with linss of ::", count)
# tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
# tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
