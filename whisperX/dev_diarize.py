import whisperx
import gc
import torch

# ------------------- Setup -------------------
audio_file = "medical_13min.mp3"  # Make sure this file exists in the same folder

# For Mac with Apple Silicon
device = "cpu"  # You can use "mps" only if align model supports it, otherwise stick with "cpu"
batch_size = 16
compute_type = "int8"  # Best for Mac if you're not using GPU

print(f"🔧 Device: {device}, Compute Type: {compute_type}")

# ------------------- Step 1: Transcription -------------------
model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)

result = model.transcribe(audio, batch_size=batch_size)
print(" Transcription Segments:")
print(result["segments"])

# ------------------- Step 2: Alignment -------------------
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device=device)
print(" Aligned Segments:")
print(result["segments"])

# ------------------- Step 3: Diarization -------------------
from whisperx.diarize import DiarizationPipeline  # ✅ Use this import on latest versions
from dotenv import load_dotenv
import os
import json
import time

# Load .env variables
load_dotenv()

# Access the token
hf_token = os.getenv("HF_TOKEN")

diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

diarize_segments = diarize_model(audio, return_embeddings=True)
speaker_embeddings = None
if isinstance(diarize_segments, tuple) and len(diarize_segments) == 2:
    diarize_segments, speaker_embeddings = diarize_segments
print("Speaker Embeddings:")
print(speaker_embeddings)
result = whisperx.assign_word_speakers(diarize_segments, result, speaker_embeddings=speaker_embeddings)

print("🎤 Diarization Segments:")
print(diarize_segments)

print("\n Final Segments with Speaker Labels:")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}: {seg['text']}")

# Save results to a TXT file in readable speaker format
txt_path = os.path.join("results", f"{os.path.splitext(audio_file)[0]}_{int(time.time())}.txt")
with open(txt_path, "w") as f:
    for seg in result["segments"]:
        speaker_label = seg['speaker']
        if isinstance(speaker_label, str) and speaker_label.startswith("SPEAKER_"):
            speaker = f"[{speaker_label}]"
        else:
            speaker = f"[SPEAKER_{int(speaker_label):02d}]"
        f.write(f"{speaker}: {seg['text'].strip()}\n")
print(f"📝 Text results saved to {txt_path}")