import time

start = time.perf_counter()
# with open("script.txt", "r") as file:
# line = file.readline()
# while line:
# text += line
# line = file.readline()


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
        audio_name = "out/bark_output_" + str(count) + ".wav"
        line = file.readline()
        if line.strip() != "" and line.__len__() > 3:
            # tts.tts_to_file(text=line, file_path=audio_name, speaker="jsmammen")
            count = count + 1
            print("line :: ", str(count), line)


end = time.perf_counter()
print(f"Execution time: {end - start:0.4f} seconds")
# Random voice generation
# tts.tts_to_file(text=text,
#                file_path="output.wav")
