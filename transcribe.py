import whisperx
import json
import torch
import os
import sys
from whisperx.diarize import DiarizationPipeline

# --- AUTOMATED INPUT & HARDWARE CONFIGURATION ---
INPUT_DIR = "data/raw_audio"
HF_TOKEN = os.getenv("HF_TOKEN")

# Automatically choose the best hardware available
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16" # Fast GPU math
elif torch.backends.mps.is_available():
    device = "mps"
    compute_type = "float32" # Apple Silicon (Mac M1/M2/M3) math
else:
    device = "cpu"
    compute_type = "int8"    # Standard CPU mode (slower, but won't crash)


# 1. Check if the input directory exists or is empty
if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
    print(f"❌ Error: The '{INPUT_DIR}' folder is empty or missing. Drop an audio file in it first!")
    sys.exit()

# 2. Automatically grab the first audio file found in the input folder
supported_extensions = ('.m4a', '.mp3', '.wav', '.mp4', '.aac')
audio_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported_extensions)]

if not audio_files:
    print(f"❌ Error: No supported audio files found in '{INPUT_DIR}' folder.")
    sys.exit()

# 3. Target the file dynamically and map names
target_file_name = audio_files[0]
audio_file = os.path.join(INPUT_DIR, target_file_name)

# 4. Generate dynamic output file name based on input file name
base_name = os.path.splitext(target_file_name)[0]  # e.g., "Recording (2)"
os.makedirs("data/diarized_transcripts", exist_ok=True)
output_json_file = f"data/diarized_transcripts/{base_name}_raw.json"

print(f"🎬 Initializing local AI engine...")
print(f"📂 Automatically detected file: {audio_file}")
print(f"💾 Output will save dynamically to: {output_json_file}")

# --- AI PIPELINE EXECUTION ---

# 1. Load Transcription Model
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# 2. Load Audio and Transcribe
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=4)

# 3. Align Timestamps
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 4. Run Speaker Diarization
print("👥 Analyzing voices and mapping speakers...")
diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
diarize_segments = diarize_model(audio)

# 5. Fuse Speakers into the Transcript
final_result = whisperx.assign_word_speakers(diarize_segments, result)

# 6. Format and Stitch Continuous Speaker Segments
print("🔄 Optimizing data structure and calculating analytics...")
stitched_segments = []
current_segment = None
speaker_durations = {}
total_duration = 0.0

for segment in final_result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
    text = segment.get("text", "").strip()
    start = round(segment.get("start", 0.0), 2)
    end = round(segment.get("end", 0.0), 2)
    
    if not text:
        continue
        
    # Calculate duration for this fragment
    duration = max(0.0, end - start)
    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
    total_duration = max(total_duration, end)
    
    # Stitch consecutive blocks from the same speaker
    if current_segment and current_segment["speaker"] == speaker:
        current_segment["text"] += f" {text}"
        current_segment["end"] = end
    else:
        if current_segment:
            stitched_segments.append(current_segment)
        current_segment = {
            "id": len(stitched_segments),
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text
        }

if current_segment:
    stitched_segments.append(current_segment)

# Clean up speaker talk times for the JSON output
for spk in speaker_durations:
    speaker_durations[spk] = round(speaker_durations[spk], 2)

# Build Production-Ready Structured Schema
structured_output = {
    "audio_metadata": {
        "file_name": target_file_name,
        "detected_language": result.get("language", "en"),
        "total_duration_seconds": round(total_duration, 2)
    },
    "conversation_analytics": {
        "total_speakers": len(speaker_durations),
        "speaker_talk_time_seconds": speaker_durations
    },
    "segments": stitched_segments
}

# 7. Write to Local JSON File
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(structured_output, f, indent=2, ensure_ascii=False)

print(f"🎉 Success! Structured data saved completely to '{output_json_file}'")