import json
import os
import sys
import shutil
import re
import ollama  # Handles the local LLM communication

print("🔒 Initializing Local LLM De-Identification Pipeline...")

# Simple platform exceptions that the LLM shouldn't count as people
PLATFORM_EXCEPTIONS = {"zoom", "canvas", "teams", "panopto", "anki", "sketchy", "jesus", "christ"}

def scrub_clinical_text(text, name_map):
    """Uses a local LLM to contextually extract names from chaotic audio 
    while natively protecting medical concepts."""
    if not text.strip():
        return text

    # Define the strict prompt for the model
    prompt = f"""
    You are a secure data redaction assistant. Read the following raw, unpunctuated audio transcript segment and extract ONLY the names of students, practitioners, or speakers.
    
    CRITICAL RULES:
    1. Do NOT extract medical diseases, eponyms (like Parkinson's, Alzheimer's, Graves), or cities/universities.
    2. Respond ONLY with a raw comma-separated list of the names found. 
    3. If no names are found, respond with the word "NONE".
    
    Transcript: "{text}"
    Names found:
    """

    try:
        # Use a lightweight, blazing-fast model (llama3 or mistral)
        response = ollama.generate(model='llama3', prompt=prompt)
        response_text = response['response'].strip()
        
        if "NONE" not in response_text.upper() and response_text:
            # Split the comma-separated names from the LLM
            extracted_names = [name.strip() for name in response_text.split(",") if name.strip()]
            
            for name in extracted_names:
                name_lower = name.lower()
                # Skip platform guardrails
                if name_lower in PLATFORM_EXCEPTIONS or len(name) <= 2:
                    continue
                
                # Register the name dynamically if we haven't seen it yet
                if name not in name_map:
                    name_map[name] = f"[STUDENT_{len(name_map) + 1}]"
                    
                # Deterministically swap out the name across this segment case-insensitively
                pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
                text = pattern.sub(name_map[name], text)
                
    except Exception as e:
        print(f"  ⚠️ LLM Processing temporary skip: {e}")
        
    return text


# ======================================================================
# 📂 MULTI-FOLDER SCANNING & ARCHIVING LOGIC (Your Exact Working Code!)
# ======================================================================
TARGET_FOLDERS = ["data/synthetic_transcripts", "data/diarized_transcripts"]
OUTPUT_CLEAN_DIR = "data/deidentified_transcripts"
ARCHIVE_DIR = "data/archive"

os.makedirs(OUTPUT_CLEAN_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

processed_count = 0

for folder in TARGET_FOLDERS:
    if not os.path.exists(folder):
        continue
        
    print(f"\n📂 Scanning directory: {folder}...")
    json_files = [f for f in os.listdir(folder) if f.lower().endswith('.json') and not f.endswith('_deidentified.json')]
    
    for json_file in json_files:
        input_json_path = os.path.join(folder, json_file)
        clean_file_name = json_file.replace(".json", "_deidentified.json")
        output_clean_path = os.path.join(OUTPUT_CLEAN_DIR, clean_file_name)
        archive_path = os.path.join(ARCHIVE_DIR, json_file)
        
        print(f"  🔄 Scrubbing file via Contextual LLM: {json_file}")
        
        try:
            with open(input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            name_registry = {}
            unique_speakers = set()
            
            for segment in data.get("segments", []):
                segment["text"] = scrub_clinical_text(segment.get("text", ""), name_registry)
                # 🏷️ Track unique raw speakers (e.g., SPEAKER_00, SPEAKER_01)
                if segment.get("speaker"):
                    unique_speakers.add(segment["speaker"])
            
            # 🛠️ Automatically inject an anonymous speaker map to satisfy his script's contract
            data["speaker_map"] = {speaker: f"Participant_{i+1}" for i, speaker in enumerate(sorted(unique_speakers))}
                
            with open(output_clean_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            shutil.move(input_json_path, archive_path)
            print(f"  📦 Archived original raw file to: {archive_path}")
            
            if name_registry:
                print(f"     Redacted Names Registry: {json.dumps(name_registry)}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed to process {json_file}. Error: {e}")

print(f"\n🎉 Done! Successfully processed and cleaned {processed_count} file(s).")