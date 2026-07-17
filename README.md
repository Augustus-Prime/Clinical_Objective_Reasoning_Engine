Markdown
# Clinical Transcript & Pedagogical Analytics Pipeline

An automated, privacy-first clinical transcription and analysis pipeline. This system accepts raw multi-speaker medical audio files, creates timestamps and diarizes speakers, scrubs Protected Health Information (PHI) entirely on a local machine, and builds structured pedagogical data logs for educational review.

---

## 🏗️ Core Pipeline Architecture
The pipeline transitions raw audio records safely through four progressive stages:

1. **Simulation Engine (`generate_mock_data.py`):** Simulates synthetic, edge-case conversation transcripts containing complex formatting and diverse PHI profiles to test downstream systems without exposure risks.
2. **Transcription & Diarization (`transcribe.py`):** Leverages **WhisperX** to isolate multi-speaker speech boundaries and outputs aligned transcripts.
3. **Local PHI De-Identification (`deidentify.py`):** Runs a local **Ollama** LLM instance to strip explicit identification markers (e.g., swapping names to `[STUDENT_1]`) directly in your hardware memory. No raw metrics leak outside.
4. **Pedagogical Structuring (`event_generation.py`):** Evaluates cleaner text through **Claude 3.5 Sonnet** to extract precise educational micro-events, timeline logs, and instructional behaviors into structured JSON formats.

---

## 🗂️ Project Directory Structure

Your project directory layout must follow this exact alignment rule:

```text
├── data/
│   ├── raw_audio/                   # Drop your source files here (*.mp3, *.m4a, *.wav)
│   ├── synthetic_transcripts/       # Populated automatically by the simulation script
│   ├── deidentified_transcripts/   # Local LLM outputs with safe speaker labels
│   └── event_transcripts/           # Final structured pedagogical JSON analytics
├──generate_mock_data.py          # Synthetic dataset simulation
├──transcribe.py         # WhisperX execution engine
├──deidentify.py         # Local Ollama scrubbing routine
├──event_generation.py      # Anthropic analytical log construction
├── requirements.txt                 # Frozen dependency configuration
└── README.md                        # Documentation guide
⚙️ Prerequisites & Infrastructure Requirements
💻 Hardware Requirements
GPU Acceleration: Running Speaker Diarization locally via WhisperX requires a dedicated NVIDIA GPU with active CUDA capabilities.

CPU Fallback: While CPU execution is technically possible, execution speeds are extremely limited. For environments lacking a strong dedicated local GPU, execute the processing chain using a cloud compute environment like Google Colab.

🔑 External Access & Secret Credentials
Before initializing runtime operations, you must secure the following accounts:

Anthropic API Key: Required for structured final logs. Set this up at console.anthropic.com.

Hugging Face Hub Token: WhisperX relies on gated community models (pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0) for speaker tracking.

Access both model cards on Hugging Face while logged in and click "Agree and access repository".

Generate an access token via huggingface.co/settings/tokens.

Create a .env file in your root folder and map your secrets:

Bash
ANTHROPIC_API_KEY="your-anthropic-key-here"
HF_TOKEN="your-huggingface-read-token-here"

🚀 Setup & Operational Installation
1. Environment & Package Installation
Initialize your execution setup by binding the exact component versions:

Bash
# Verify your local environment links to a valid Python installation, then install requirements:
pip install -r requirements.txt
Create a file named requirements.txt containing these explicit pins:

Plaintext
anthropic==0.116.0
ollama==0.6.2
torch==2.12.1
whisperx==3.8.6

2. Configure Your Local Machine for Anonymization
Download and launch your target local framework instance to keep patient assets enclosed safely within local storage:

Bash
# 1. Download Ollama from [https://ollama.com](https://ollama.com)
# 2. Open your terminal shell and retrieve the Llama3 engine locally:
ollama run llama3
🏃 Execution Commands
Full Pipeline Run (Recommended)
Launch the pipeline controller to run all scripts interactively from end-to-end:


Executing Separate Component Scripts
If you want to run stages manually step-by-step:

Bash
# Stage 1A: Generate Mock Test Materials
python generate_mock_data.py

# Stage 2B: Audio Conversion, Timestamps, and Speaker Tracking
python transcribe.py

# Stage 3: Local PHI Scrubbing with Ollama
python deidentify.py

# Stage 4: Anthropic Deep Pedagogical Analytics Engine
python event_generation.py

📊 Expected Data Outflows
Upon operational completion, verify structural pipeline delivery targets across your directories:

data/deidentified_transcripts/*_deidentified.json: Contains sanitized conversation trees containing unified speaker assignments (Participant_1, Participant_2) free of identity footprints.

data/event_transcripts/*_events.json: Delivers an instructional matrix featuring categorized micro-events, teaching behavior breakdowns, and timeline indices.
