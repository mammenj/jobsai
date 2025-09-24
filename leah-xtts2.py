import torch
from TTS.api import TTS
import time
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

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # with open("script.txt", "r") as file:
    # line = file.readline()
    # while line:
    # text += line
    # line = file.readline()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    count = 0
    with open("script.txt", "r") as file:
        line = file.readline()
        print("Starting processing...", line)
        while line:
            # text += line
            print("while Processing line: ", line)
            count = count + 1
            # audio_name = getName(line)
            # audio_path = f"out/xtts2/{count:03d}-{audio_name}-leah.wav"
            audio_path = f"out/leah/leah-{count:03d}.wav"

            if line.strip() != "":  # Only process non-empty lines
                print("Generating ", audio_path)
                tts.tts_to_file(
                    text=line,
                    file_path=audio_path,
                    speaker="leah",
                    language="en",
                    split_sentences=False,
                )
                print("Done with ", audio_path)

    end = time.perf_counter()
    print(f"Execution time: {end - start:0.4f} secondsi, count: {count:03d}")


if __name__ == "__main__":
    main()
