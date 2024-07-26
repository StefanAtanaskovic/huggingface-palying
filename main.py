from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import pygame


print("staring generation of text")
checkpoint = "HuggingFaceTB/SmolLM-1.7B"
device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

question = input("Enter you question: ")

# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
inputs = tokenizer.encode(question, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=100)
text = tokenizer.decode(outputs[0])


print("staring generation of speach")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load("speech.wav")

# Play the audio file
pygame.mixer.music.play()

# Keep the program running until the audio finishes
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
