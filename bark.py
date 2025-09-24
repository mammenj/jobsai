import time

import torch
from TTS.api import TTS
import re


def getName(sentence):
    clean_sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

    words = clean_sentence.split()
    if words:
        firstword = words[0]
        lastword = words[-1]
        return firstword + "_" + lastword


def main():
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text1 = "[clears throat] Hi my name is John MAMMEN [sighs] -  I am a software developer. I love coding in Python.[laughs]"
    text = ""
    # with open("script.txt", "r") as file:
    # line = file.readline()
    # while line:
    # text += line
    # line = file.readline()

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
    count = 0
    with open("script.txt", "r") as file:
        line = file.readline()
        while line:
            # text += line
            count = count + 1
            line = file.readline()
            if line.strip() != "" and line.__len__() > 3:
                print("Processing :: ", line)
                audio_name = getName(line)
                audio_path = f"out/bark/{count:03d}-{audio_name}.wav"
                tts.tts_to_file(text=line, file_path=audio_path, speaker="jsmammen")
                print("Done with ", audio_name)

    end = time.perf_counter()
    print(f"Execution time: {end - start:0.4f} seconds")


if __name__ == "__main__":
    main()
