import torch
from TTS.api import TTS
import time
import re

start = time.perf_counter()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


def getName(sentence):
    clean_sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

    words = clean_sentence.split()
    if words:
        firstword = words[0]
        lastword = words[-1]
        return firstword + "_" + lastword


def main():
    # with open("script.txt", "r") as file:
    # line = file.readline()
    # while line:
    # text += line
    # line = file.readline()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    count = 0
    with open("script.txt", "r") as file:
        line = file.readline()
        while line:
            # text += line
            print("Processing line: ", line)
            count = count + 1
            audio_name = getName(line)
            audio_path = f"out/xtts2/{count:03d}-{audio_name}.wav"

            line = file.readline()
            if line.strip() != "" and line.__len__() > 3:
                tts.tts_to_file(
                    text=line,
                    file_path=audio_path,
                    speaker="newjsmammen",
                    language="en",
                    split_sentences=False,
                )
                print("Done with ", audio_name)

    end = time.perf_counter()
    print(f"Execution time: {end - start:0.4f} seconds")


if __name__ == "__main__":
    main()
