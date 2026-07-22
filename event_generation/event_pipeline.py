#!/usr/bin/env python3
"""
Master Event Pipeline
Location: event_generation/event_pipeline.py

Orchestrates the entire flow in memory:
  Raw Transcript JSON -> Boundary Detection -> Stage 1 (GLiNER) -> Stage 2 (Ollama) -> Final Intake JSON
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is at the very front of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_generation.boundary_detector import group_transcript_into_exchanges
from event_generation.event_analyzer import analyze_exchange_cognition
from event_extractor import extract_stage_1_entities


def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS formatted string."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def sanitize_analysis(analysis: dict) -> dict:
    """Converts pseudo-null strings from LLM into clean JSON types."""
    if not isinstance(analysis, dict):
        return {}
    for key, val in analysis.items():
        if isinstance(val, str) and val.strip().lower() in ["none", "n/a", "null", ""]:
            analysis[key] = None if key != "cognitive_trajectory" else "Stable Understanding"
    return analysis


def get_first_deidentified_transcript() -> Path:
    """Pulls the first JSON file in the stack from data/deidentified_transcripts."""
    folder = PROJECT_ROOT / "data" / "deidentified_transcripts"
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    files = sorted(list(folder.glob("*.json")))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")
        
    return files[0]


def run_pipeline(input_json_path: str = None, output_json_path: str = None) -> Dict[str, Any]:
    """Runs the 3-step pipeline on a raw transcript file."""
    
    # 1. Resolve Input File Path
    if input_json_path:
        path = Path(input_json_path)
    else:
        path = get_first_deidentified_transcript()
        print(f"[INFO] No input specified. Pulled first of stack: '{path.name}'")

    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    # 2. Resolve Output File Path (Default: data/event_transcripts/<stem>_events.json)
    if not output_json_path:
        out_dir = PROJECT_ROOT / "data" / "event_transcripts"
        output_json_path = out_dir / f"{path.stem}_events.json"

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_segments = raw_data.get("segments", [])
    print(f"[INFO] Loaded {len(raw_segments)} raw segments from '{path.name}'.")

    # STEP 1: Boundary Detection
    print("[STEP 1] Grouping segments into dialogue exchanges...")
    exchanges = group_transcript_into_exchanges(raw_segments)
    print(f"         Created {len(exchanges)} exchange blocks.")

    # State tracking across session
    active_patients = []
    generated_events = []

    # Process each exchange through Stage 1 & Stage 2
    for idx, ex in enumerate(exchanges, 1):
        event_id = f"evt_{idx:04d}"
        dialogue_text = ex["dialogue_block"]
        participants = ex["all_participants"]
        primary_speaker = ex["primary_speaker"]
        responders = [p for p in participants if p != primary_speaker]

        start_str = format_timestamp(ex["time_window"]["start"])
        end_str = format_timestamp(ex["time_window"]["end"])
        timestamp_range = f"{start_str} - {end_str}"

        print(f"\n[STEP 2 & 3] Processing {event_id} ({timestamp_range})...")

        # STEP 2: Stage 1 GLiNER Entity Extraction
        entities = extract_stage_1_entities(dialogue_text)
        extracted_patients = entities.get("patients", [])
        clinical_terms = entities.get("clinical_terms", [])

        # Patient Context Persistence
        if extracted_patients:
            active_patients = extracted_patients
        
        effective_patients = extracted_patients if extracted_patients else active_patients

        # STEP 3: Stage 2 Ollama Cognitive Assessment
        llm_result = analyze_exchange_cognition(dialogue_text, clinical_terms)
        
        # Sanitize LLM output strings (e.g. "None" -> null)
        clean_analysis = sanitize_analysis(llm_result.get("analysis", {}))

        # Assemble full intake-compatible event payload
        event_payload = {
            "event_id": event_id,
            "timestamp_range": timestamp_range,
            "content_class": "ACADEMIC",
            "event_type": "concept_building",
            "primary_speaker": primary_speaker,
            "responders": responders,
            "case_entities": {
                "patients": effective_patients,
                "eponyms_and_medical_terms": clinical_terms
            },
            "summary": llm_result.get("summary", ""),
            "transcript_quote": dialogue_text,
            "analysis": clean_analysis
        }

        generated_events.append(event_payload)

    final_output = {
        "source_file": path.name,
        "total_events": len(generated_events),
        "events": generated_events
    }

    # Save to disk
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[SUCCESS] Pipeline complete! Output saved to: {out_path}")
    return final_output


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Event Generation Pipeline")
    # Make input optional: if omitted, pulls first from stack
    parser.add_argument("input_json", nargs="?", help="Path to input transcript JSON (optional)", default=None)
    parser.add_argument("-o", "--output", help="Path to save output JSON (optional)", default=None)
    args = parser.parse_args()

    try:
        run_pipeline(args.input_json, args.output)
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)