#!/usr/bin/env python3
"""
Stage 2 Analyzer Module
Location: event_generation/stage2_analyzer.py

Uses Ollama to generate full intake-compatible analysis objects, including
both legacy tags (confidence_level, affective_tone) and rich faculty insights.
"""

import re
import json
import sys
import urllib.request
import urllib.error
from typing import Dict, List, Any

OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

STAGE_2_PROMPT_TEMPLATE = """You are an expert medical educator evaluating a clinical group discussion.

[CLINICAL TERMS EXTRACTED]
{clinical_terms}

[DIALOGUE EXCHANGE BLOCK]
{dialogue_block}

Analyze the exchange and output a SINGLE RAW JSON object matching this exact schema:

{{
  "summary": "<1-2 sentence objective summary of what was discussed>",
  "analysis": {{
    "interaction_type": "<'Peer-to-Peer' or 'Instructor-Led'>",
    "discussion_purpose": "<'Concept Building', 'Differential Diagnosis', or 'Treatment Planning'>",
    "cognitive_state": "<'Mastery', 'Partial Understanding', 'Misconception', or 'Complete Confusion'>",
    "understanding_score": <int 1-5>,
    "confidence_level": "<'low', 'moderate', or 'high'>",
    "affective_tone": "<'Neutral', 'Inquisitive', 'Hesitant', or 'Confident'>",
    "evidence_strength": "<'weak', 'moderate', or 'strong'>",
    "cognitive_trajectory": "<MUST BE A NON-EMPTY STRING, e.g. 'Initial Misconception -> Peer Correction -> Mastery' or 'Consistent Mastery'>",
    "key_misconception": "<Explicitly describe any medical error or anchoring bias mentioned. Write 'None' if no error occurred. DO NOT RETURN NULL.>",
    "reasoning_breakthrough": "<Detail how peers or discussion resolved a misconception, or 'None' if not applicable>",
    "faculty_summary": "<1-2 sentence actionable takeaway for the instructor. MUST NOT BE EMPTY.>"
  }}
}}

CRITICAL INSTRUCTIONS:
1. If a student incorrectly anchors on a lab value (e.g. troponin = automatic ACS), you MUST log this in 'key_misconception'.
2. NEVER leave 'cognitive_trajectory' or 'faculty_summary' as empty strings or null.
3. Return ONLY raw JSON without markdown code blocks.
"""
def parse_json_safely(raw_response: str) -> Dict[str, Any]:
    """Extracts valid JSON from raw LLM output even if surrounded by markdown or commentary."""
    # Attempt 1: Direct JSON load
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Use Regex to extract outer {...} braces
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # If both fail, raise exception to trigger the fallback response
    raise json.JSONDecodeError("Could not extract valid JSON from response", raw_response, 0)
def analyze_exchange_cognition(
    dialogue_block: str, 
    clinical_terms: List[str], 
    model: str = OLLAMA_MODEL
) -> Dict[str, Any]:
    """Sends exchange text to Ollama and returns full analysis payload."""
    terms_str = ", ".join(clinical_terms) if clinical_terms else "None extracted"
    prompt = STAGE_2_PROMPT_TEMPLATE.format(
        clinical_terms=terms_str,
        dialogue_block=dialogue_block
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_response = res_data.get("response", "{}")
            return parse_json_safely(raw_response)

    except urllib.error.URLError as e:
        print(f"[ERROR] Could not connect to Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        return _fallback_response("Ollama connection error")
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse JSON response from Ollama.", file=sys.stderr)
        return _fallback_response("JSON parsing error")

def _fallback_response(error_reason: str) -> Dict[str, Any]:
    return {
        "summary": "Exchange analysis failed.",
        "analysis": {
            "interaction_type": "Peer-to-Peer",
            "discussion_purpose": "Concept Building",
            "cognitive_state": "Partial Understanding",
            "understanding_score": 3,
            "confidence_level": "moderate",
            "affective_tone": "Neutral",
            "evidence_strength": "moderate",
            "cognitive_trajectory": "Unevaluated",
            "key_misconception": None,
            "reasoning_breakthrough": None,
            "faculty_summary": f"Error during analysis: {error_reason}"
        }
    }

if __name__ == "__main__":
    sample_dialogue = (
        'SPEAKER_02: "Look at the troponin levels. They\'re elevated... looks like acute coronary syndrome to me."\n'
        'SPEAKER_03: "Wait, the patient\'s been symptomatic for months. This isn\'t ACS as the primary diagnosis."\n'
        'SPEAKER_01: "Troponin can be elevated in heart failure itself from chronic strain. It doesn\'t automatically mean acute MI."'
    )
    sample_terms = ["acute coronary syndrome", "troponin", "heart failure", "acute MI"]

    print("\n[INFO] Testing Full Intake Stage 2 Analyzer...")
    result = analyze_exchange_cognition(sample_dialogue, sample_terms)
    print("\n" + "="*60)
    print("STAGE 2 ANALYSIS RESULT (FULL INTAKE SCHEMA)")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60 + "\n")