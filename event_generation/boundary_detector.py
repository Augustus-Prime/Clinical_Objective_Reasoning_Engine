#!/usr/bin/env python3
"""
Boundary Detector Module
Location: event_generation/boundary_detector.py

Groups raw transcript segments into dialogue exchange events using a hybrid approach:
word-count safety bounds + asymmetry-safe lexical overlap tracking.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Any


class HybridBoundaryDetector:
    """Combines structural word caps with fixed (asymmetry-safe) lexical overlap."""
    
    def __init__(self, overlap_threshold: float = 0.20, min_words: int = 50, max_words: int = 220):
        self.overlap_threshold = overlap_threshold
        self.min_words = min_words
        self.max_words = max_words
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "so", "is", "are", "was", "were", 
            "we", "you", "they", "he", "she", "it", "this", "that", "to", "of", "in", 
            "um", "uh", "like", "yeah", "okay", "right", "mean", "got"
        }

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return {w for w in words if w not in self.stopwords}

    def should_split(self, current_block: str, next_line: str, current_word_count: int) -> bool:
        # Guard 1: Don't split tiny events (need minimal context for downstream analysis)
        if current_word_count < self.min_words:
            return False
            
        # Guard 2: Hard cap to keep events focused
        if current_word_count >= self.max_words:
            return True

        block_tokens = self._tokenize(current_block)
        line_tokens = self._tokenize(next_line)

        if not block_tokens or not line_tokens:
            return False

        # Measure overlap relative to incoming line length (fixes asymmetry bug)
        intersection = block_tokens.intersection(line_tokens)
        overlap_ratio = len(intersection) / len(line_tokens)

        return overlap_ratio < self.overlap_threshold


def group_transcript_into_exchanges(
    segments: List[Dict[str, Any]], 
    overlap_threshold: float = 0.20,
    min_words: int = 50,
    max_words: int = 220
) -> List[Dict[str, Any]]:
    """Groups raw transcript segments into cohesive multi-turn dialogue exchanges."""
    detector = HybridBoundaryDetector(
        overlap_threshold=overlap_threshold,
        min_words=min_words,
        max_words=max_words
    )

    grouped_exchanges = []
    current_segments = []
    current_word_count = 0
    current_block_text = ""
    exchange_id = 1

    for seg in segments:
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "UNKNOWN")
        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", 0.0)

        if not text:
            continue

        word_count = len(text.split())

        # Check boundary break before adding segment
        if current_segments and detector.should_split(current_block_text, text, current_word_count):
            grouped_exchanges.append(_build_exchange_payload(exchange_id, current_segments))
            exchange_id += 1
            current_segments = []
            current_word_count = 0
            current_block_text = ""

        current_segments.append({
            "speaker": speaker,
            "text": text,
            "start": start_time,
            "end": end_time
        })
        current_word_count += word_count
        current_block_text += " " + text

    # Flush remaining segments
    if current_segments:
        grouped_exchanges.append(_build_exchange_payload(exchange_id, current_segments))

    return grouped_exchanges


def _build_exchange_payload(exchange_id: int, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Formats a cluster of segments into a single exchange block."""
    raw_dialogue_lines = [f"{s['speaker']}: \"{s['text']}\"" for s in segments]
    combined_dialogue = "\n".join(raw_dialogue_lines)
    
    unique_speakers = list(dict.fromkeys([s["speaker"] for s in segments]))
    start_time = segments[0]["start"]
    end_time = segments[-1]["end"]
    
    return {
        "exchange_id": f"EXCHANGE_{exchange_id:03d}",
        "time_window": {"start": start_time, "end": end_time},
        "primary_speaker": unique_speakers[0] if unique_speakers else "UNKNOWN",
        "all_participants": unique_speakers,
        "raw_turn_count": len(segments),
        "dialogue_block": combined_dialogue
    }


# =====================================================================
# CLI EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a transcript JSON file path.")
        print("Usage: python event_generation/boundary_detector.py <path_to_transcript.json>")
        sys.exit(1)
        
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"[ERROR] File '{file_path}' not found.")
        sys.exit(1)
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    raw_segments = data.get("segments", [])
    print(f"\n[INFO] Ingesting {len(raw_segments)} raw segments from '{file_path.name}'...")
    
    exchanges = group_transcript_into_exchanges(raw_segments)
    
    print("\n" + "="*60)
    print(f"HYBRID BOUNDARY DETECTOR: {len(exchanges)} EXCHANGES CREATED")
    print("="*60 + "\n")
    
    for ex in exchanges:
        print(f"📍 {ex['exchange_id']} (Turns: {ex['raw_turn_count']}, Speakers: {ex['all_participants']})")
        print(f"   Time Window: {ex['time_window']['start']}s - {ex['time_window']['end']}s")
        print("   Dialogue Block:")
        for line in ex['dialogue_block'].split("\n"):
            print(f"     {line}")
        print("-" * 60)