from TTS.api import TTS

text1 = """    We talked of the side show in the circus.    
    Use api pencil to write the first draft.    He ran half way to the hardware store.    The clock struck to mark the third period.    
    A small creek cutacross the field.    Cars and busses stalled in snow drifts.    The set of china hit the floor with a crash.    
This is a grand season for hikes on the road.    The dune rose from the edge of the water.   Those words were the cue for the actor to leave."""


text = ""
with open("script.txt", "r") as file:
    line = file.readline()
    while line:
        text += line
        line = file.readline()


# Get device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# Init TTS with the target model name
# tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(
#    device
# )

# Run TTS
# tts.tts_to_file(text=text, file_path="./test")
tts = TTS(model_name="tts_models/en/multi-dataset/tortoise-v2")
tts.tts_to_file(
    text1,
    speaker_wav=["voicedir/myvoice.wav"],
    speaker="jsmammen",
    file_path="out/tortise-v2-output.wav",
    num_autoaggressive_samples=1,
    diffusion_iterations=10,
)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/en/multi-dataset/tortise-v2")
tts.tts_to_file(
    text1,
    speaker="jsmammen",
    file_path="out/tortise-v2-output.wav",
    preset="ultra_fast",
)
print("Done.......")
# tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
# tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
