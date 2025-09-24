import torch
from TTS.api import TTS
import time
import re

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
#
def getName(sentence):
    clean_sentence = re.sub(r"[^a-zA-Z0-9 ]", "", sentence)

    words = clean_sentence.split()
    if words:
        firstword = words[0]
        lastword = words[-1]
        return firstword + "_" + lastword


def main():
    print("Starting TTS with voice cloning...tts_models/en/ljspeech/fast_pitch")
    tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)

    if train:
        tts.tts_to_file(
            text="This is voice cloning. I am john mammen [laughs] and I am testing yourtts model [sighs]",
            speaker_wav="voicedir/jsmammen.wav",
            speaker="jsmammen",
            language="en",
            file_path="out/fast_pitch2/fast_pitch2-output.wav",
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
                audio_name = getName(line)
                audio_path = f"out/fastpitch2/{count:03d}-{audio_name}.wav"
                tts.tts_to_file(
                    line,
                    # speaker="ljspeech",
                    speaker_wav="voicedir/jsmammen.wav",
                    # language="en",
                    file_path=audio_path,
                )
                end = time.perf_counter()
                print(
                    f"Time taken to process {count} :: {end - start:0.4f} seconds :: for {audio_path}"
                )

    print("TTS with voice cloning completed with linss of ::", count)


if __name__ == "__main__":
    main()
