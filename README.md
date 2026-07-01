Markdown
# CLINICAL OBJECTIVE REASONING ENGINE (CORE) — Local Pipeline

This repository contains the local, privacy-first transcription and de-identification pipeline built for the AI-Informatics Curriculum Efficiency Mapping project.

## 🔑 Authentication (Hugging Face Token)
This pipeline utilizes Pyannote for speaker diarization, which requires an access token.
1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Accept the user terms for `pyannote/speaker-diarization-community` and `pyannote/segmentation-3.0`.
3. Generate a Read Token in your Hugging Face Account Settings.
4. Pass your token to the terminal environment when running the script (see execution steps below).

## 🛠️ Prerequisites
Ensure you have **Python 3.10, 3.11, or 3.12** installed on your machine.

## 🚀 Local Environment Setup

1. **Create your local Virtual Environment:**
   ```bash
   python -m venv venv
   ```
2. **Activate the environment:**

Windows (PowerShell): 
```powershell
.\venv\Scripts\Activate.ps1
```

Mac / Linux: 
```bash
source venv/bin/activate
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```
Download the local NLP Text Scanner model:

```bash
python -m spacy download en_core_web_sm
```

## 📂 Data Execution Flow
1. Create a folder named input in the root directory.

2. Drop any raw audio file (.m4a, .mp3, .wav) into the input/ folder.

3. Run the GPU-adaptive transcription script (injecting your token):
   * **Windows:** `$env:HF_TOKEN="your_token_here"; python transcribe.py`
   * **Mac/Linux:** `export HF_TOKEN="your_token_here"; python transcribe.py`
   
(Generates [YourFileName]_raw.json)

4. Run the local HIPAA de-identification script:

```bash
python deidentify.py
```
(Generates [YourFileName]_deidentified.json)

## 🧠 Hardware Compatibility
The transcribe.py script automatically scales to your hardware:

NVIDIA GPU: Runs lightning-fast via CUDA (float16).

Apple Silicon (M1/M2/M3 Mac): Automatically scales to MPS framework.

Standard CPU: Fallback execution layer (slower, but completely stable).
