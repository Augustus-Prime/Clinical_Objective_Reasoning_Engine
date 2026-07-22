#!/usr/bin/env python3
"""
Stage 1 Extractor: Zero-Shot Entity Extraction via GLiNER.
Can be imported as a module OR run directly on a transcript JSON file.

Usage:
    python stage1_extractor.py path/to/transcript.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
from gliner import GLiNER

# 1. Target labels tailored for clinical/academic transcripts
TARGET_LABELS = [
    "patient demographic",
    "medical condition",
    "symptom",
    "laboratory test or value",
    "medication or drug",
    "procedure or diagnostic imaging",
    "anatomical structure"
]

GENERIC_PATIENT_NOISE = {
    "patient", "the patient", "she", "he", "they", "him", "her", 
    "someone", "guy", "woman", "man", "individual", "person"
}

# 2. Global model lazy-loading (loads once when needed)
_GLINER_MODEL = None

def get_gliner_model():
    global _GLINER_MODEL
    if _GLINER_MODEL is None:
        print("[INFO] Loading GLiNER zero-shot model...", file=sys.stderr)
        _GLINER_MODEL = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    return _GLINER_MODEL


def clean_patient_entities(raw_patients: List[str]) -> List[str]:
    """Filters out speaker tags, pronouns, and vague terms from patient entities."""
    clean_list = []
    for item in raw_patients:
        text = item.strip()
        if re.match(r'^SPEAKER_\d+$', text, re.IGNORECASE):
            continue
        if text.lower() in GENERIC_PATIENT_NOISE:
            continue
        if len(text) < 3:
            continue
        clean_list.append(text)
    return list(set(clean_list))


def extract_stage_1_entities(text: str, threshold: float = 0.35) -> Dict[str, List[str]]:
    """
    Core Stage 1 Function.
    Takes raw transcript text and returns clean patient and clinical term lists.
    """
    model = get_gliner_model()
    entities = model.predict_entities(text, TARGET_LABELS, threshold=threshold)
    
    raw_patients = []
    clinical_terms = []
    
    for ent in entities:
        label = ent["label"]
        span_text = ent["text"].strip()
        
        if label == "patient demographic":
            raw_patients.append(span_text)
        else:
            if len(span_text) > 1:
                clinical_terms.append(span_text)
                
    dedup_terms = sorted(list({t.lower(): t for t in clinical_terms}.values()))
    
    return {
        "patients": clean_patient_entities(raw_patients),
        "clinical_terms": dedup_terms
    }


def process_transcript_file(file_path: str) -> List[Dict[str, Any]]:
    """Ingests a raw transcript JSON and extracts entities for every segment."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    segments = data.get("segments", [])
    processed_results = []
    
    print(f"[INFO] Ingesting {len(segments)} segments from '{path.name}'...", file=sys.stderr)
    
    for idx, seg in enumerate(segments, 1):
        quote = seg.get("text", "").strip()
        speaker = seg.get("speaker", f"SPEAKER_{idx:02d}")
        
        if not quote:
            continue
            
        entities = extract_stage_1_entities(quote)
        
        processed_results.append({
            "segment_id": idx,
            "speaker": speaker,
            "text_quote": quote,
            "case_entities": entities
        })
        
    return processed_results


# =====================================================================
# CLI INGESTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Entity Extractor CLI")
    parser.add_argument("input_json", help="Path to raw transcript JSON file")
    parser.add_argument("-o", "--output", help="Optional path to save Stage 1 output JSON")
    args = parser.parse_args()
    
    try:
        results = process_transcript_file(args.input_json)
        
        # If output file specified, write JSON; otherwise print formatted results
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[SUCCESS] Saved Stage 1 extractions to: {out_path}")
        else:
            print("\n" + "="*60)
            print("STAGE 1 TRANSCRIPT EXTRACTION RESULTS")
            print("="*60 + "\n")
            for res in results[:5]:  # Preview first 5 segments
                print(f"Segment #{res['segment_id']} [{res['speaker']}]:")
                print(f"  Quote:    \"{res['text_quote']}\"")
                print(f"  Patients: {res['case_entities']['patients']}")
                print(f"  Terms:    {res['case_entities']['clinical_terms']}")
                print("-" * 60)
            if len(results) > 5:
                print(f"... showing 5 of {len(results)} segments total.")
                
    except Exception as e:
        print(f"[ERROR] Failed to process file: {e}", file=sys.stderr)
        sys.exit(1)