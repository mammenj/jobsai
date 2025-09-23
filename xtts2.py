import torch
from TTS.api import TTS
import time

start = time.perf_counter()


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


count = 0
with open("script.txt", "r") as file:
    line = file.readline()
    while line:
        # text += line
        print("Processing line: ", line)
        count = count + 1
        audio_name = "out/xtts_output_" + str(count) + ".wav"
        line = file.readline()
        if line.strip() != "" and line.__len__() > 3:
            tts.tts_to_file(
                text=line,
                file_path=audio_name,
                speaker="jsmammen",
                language="en",
                split_sentences=False,
            )
            print("Done with ", audio_name)


end = time.perf_counter()
print(f"Execution time: {end - start:0.4f} seconds")
