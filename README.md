
# 🎙️ WhisperX Diarization Pipeline on macOS (Apple Silicon)

This project sets up and runs **WhisperX** for speaker diarization on an audio file (`Catching.mp3`) using a developer-friendly script with Hugging Face authentication secured via `.env`.

---

## 📦 Requirements

- Python 3.9+
- MacBook with Apple Silicon (M1/M2/M3 or newer)
- `ffmpeg` installed
- WhisperX and dependencies
- Hugging Face account and access token

---

## 🛠️ Setup Instructions

### 1. Clone This Repository

```bash
git clone https://github.com/ayushdh96/Pyannote_Diarization.git
cd Pyannote_Diarization/whisperX
```

This will bring you into the WhisperX folder inside the cloned project.

---

### 2. Install dependencies

Install WhisperX in developer mode:

```bash
pip install -e .
```

Install Python environment utilities:

```bash
pip install python-dotenv
```

---

### 3. Set up `.env` with your Hugging Face token

Create a `.env` file in the **same folder as `dev_diarize.py`**:

```bash
touch .env
```

Paste this inside `.env`:

```
HF_TOKEN=your_huggingface_token_here
```

> 🔒 **Keep your `.env` file private** and don't share it or push it to GitHub.

---

### 4. Add your audio file

Place your audio file (`Catching.mp3`) in the same directory as `dev_diarize.py`.

---

## ▶️ Run the Diarization Script

```bash
python dev_diarize.py
```

You should see output like:

```
📌 Step 1: Transcribing...
✅ Transcription complete.
📌 Step 2: Word-level alignment...
✅ Alignment complete.
📌 Step 3: Speaker diarization...
✅ Diarization complete.
🎤 Final diarized output:
[0.67s - 2.45s] Speaker 0: Hello Mark...
...
💾 Saved to diarized_output.json
```

---

## 📁 Output

- `diarized_output.json`: Contains word-level timestamps and assigned speaker IDs.

---

## ❓ Notes

- This uses `device="cpu"` for Apple Silicon compatibility. You can try `"mps"` but alignment may fail.
- Ensure your Hugging Face account has accepted the usage conditions for `pyannote/speaker-diarization-*`.

- Additionally, visit the following Hugging Face model pages and click **"Access Repository"** to grant yourself permission to use the gated models:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> 📌 You must be logged into your Hugging Face account and click the **"Agree and Access Repository"** button on each model page to avoid runtime download errors.

---

## 📬 Credits

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [Pyannote](https://huggingface.co/pyannote)
