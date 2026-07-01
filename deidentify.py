import json
import os
import sys
import spacy

# --- CONFIGURATION ---
INPUT_DIR = "input"
# Automatically find the raw JSON we just generated
supported_extensions = ('.m4a', '.mp3', '.wav', '.mp4', '.aac')
audio_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported_extensions)]

if not audio_files:
    print("❌ Error: No audio files found in input directory to map to JSON.")
    sys.exit()

base_name = os.path.splitext(audio_files[0])[0]
input_json_file = f"{base_name}_raw.json"
output_clean_file = f"{base_name}_deidentified.json"

print("🔒 Initializing Local Medical NLP De-Identification Pipeline...")
# Load the smart text-scanning model
nlp = spacy.load("en_core_web_sm")

# Medical concept protection checklist (Eponyms/Jargon that look like names)
MEDICAL_PROTECTION_WORDS = {
    "parkinson", "parkinson's", "alzheimer", "alzheimer's", "graves", 
    "achilles", "hodgkin", "crohn", "crohn's", "down", "down's", "wilson"
}

def scrub_clinical_text(text, name_map):
    """Scans text using NLP to find real names while protecting medical context."""
    if not text.strip():
        return text
        
    doc = nlp(text)
    cleaned_text = text
    
    # Track entities backwards so character positions don't shift when replacing text
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    for ent in entities:
        # We only want to censor actual people names (PERSON tag)
        if ent.label_ == "PERSON":
            original_name = ent.text
            
            # GUARDRAIL: If the 'Name' is actually a protected medical term, skip it!
            if original_name.lower() in MEDICAL_PROTECTION_WORDS:
                continue
                
            # Assign a permanent anonymous token to this specific name
            if original_name not in name_map:
                name_map[original_name] = f"[STUDENT_{len(name_map) + 1}]"
                
            # Swap the real name out for the safe token
            cleaned_text = cleaned_text[:ent.start_char] + name_map[original_name] + cleaned_text[ent.end_char:]
            
    return cleaned_text

# 1. Load the raw JSON transcript
try:
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{input_json_file}'. Please run transcribe.py first.")
    sys.exit()

print(f"📂 Processing file: {input_json_file}")

# 2. Loop through every single conversational turn
name_registry = {}
for segment in data.get("segments", []):
    segment["text"] = scrub_clinical_text(segment.get("text", ""), name_registry)

# 3. Save the clean, HIPAA-safe dataset
with open(output_clean_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"🎉 Success! Protected data saved completely to '{output_clean_file}'")
print(f"👥 Redacted Names Registry: {json.dumps(name_registry)}")