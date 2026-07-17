#!/usr/bin/env python3
"""Convert a diarized transcript JSON into a structured conversational-events JSON.

Uses the Anthropic API to segment a diarized transcript into conversational
"events", merging consecutive segments that form a single conversational move
and labeling each event with a content class, event type, extracted medical
entities, and (for non-noise events) a pedagogical analysis.

Usage:
    python transcript_to_events.py input.json -o output.json
"""

from __future__ import annotations

import argparse
from email import parser
from email import parser
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192

# Bundled few-shot pair (input transcript + its gold-standard events output).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXAMPLE_INPUT = SCRIPT_DIR / "session1.json"
DEFAULT_EXAMPLE_OUTPUT = SCRIPT_DIR / "session12.json"

CONTENT_CLASSES = {"ACADEMIC", "NOISE", "AFFECT"}
EVENT_TYPES = {
    "logistics",
    "case_presentation",
    "concept_demonstration",
    "question",
    "peer_correction",
    "receptive_uptake",
}
ANALYSIS_FIELDS = [
    "interaction_type",
    "discussion_purpose",
    "cognitive_state",
    "understanding_score",
    "confidence_level",
    "affective_tone",
    "evidence_strength",
]
EVENT_ID_RE = re.compile(r"^evt_\d{4}$")
TIMESTAMP_RANGE_RE = re.compile(r"^\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}$")

SYSTEM_PROMPT = """\
You are an expert analyst of medical-education small-group discussions. You are
given a diarized transcript (speaker-labeled segments with float timestamps) and
must convert it into a structured JSON object of conversational EVENTS.

# Output contract

Return a single JSON object with exactly one top-level key, "events", whose value
is an array of event objects. Output JSON ONLY — no prose, no markdown fences.

# Merging segments into events

Consecutive segments that together form ONE conversational move must be MERGED
into a single event. A "move" is a coherent unit such as:
  - a question plus the answer it directly elicits,
  - an explanation plus a peer's immediate elaboration or affirmation,
  - a logistics exchange (e.g. a request and its reply).
When you merge, the first substantive speaker is the primary_speaker and the
other contributors are listed in responders (in order, human labels, no repeats).
Do NOT merge across a genuine topic/agenda shift.

# Event schema

Every event object has these fields, in this order:
  - event_id: string, sequential, zero-padded: "evt_0001", "evt_0002", ...
  - timestamp_range: string "HH:MM:SS - HH:MM:SS". Convert the float seconds of
    the merged span into clock time. Start = start of the first merged segment;
    end = end of the last merged segment (round to whole seconds).
  - content_class: one of "ACADEMIC", "NOISE", "AFFECT".
      ACADEMIC = on-topic medical/scientific content.
      NOISE    = off-topic chatter or pure logistics (chargers, agenda, mixers).
      AFFECT   = primarily an emotional/attitudinal reaction (excitement,
                 appreciation) rather than new academic content.
  - event_type: one of "logistics", "case_presentation", "concept_demonstration",
    "question", "peer_correction", "receptive_uptake".
  - primary_speaker: human label (from speaker_map) of the event's main speaker.
  - responders: array of human labels who responded/elaborated (may be empty).
  - case_entities: object with two arrays:
      - patients: patient descriptors mentioned (e.g. "2-year-old male"), else [].
      - eponyms_and_medical_terms: medical/scientific terms mentioned, in order of
        appearance, de-duplicated. STRIP LaTeX: "$CO_2$" -> "CO2",
        "$NAD^+$" -> "NAD+", "$Ca^{2+}$" -> "Ca2+".
  - linked_event_id: the event_id this event directly responds to (e.g. an answer
    links to its question, a correction to the claim it corrects), or null.
  - summary: one concise sentence describing what happens in the event.
  - analysis: OBJECT — INCLUDE for ACADEMIC and AFFECT events; OMIT ENTIRELY for
    NOISE events. When present it has exactly these fields:
      - interaction_type: short phrase describing the interaction.
      - discussion_purpose: short phrase describing the pedagogical purpose.
      - cognitive_state: short phrase describing the speaker's cognitive state.
      - understanding_score: integer 0-5 (0 = no content-understanding
        demonstrated, e.g. a teacher prompt or pure affect; 5 = strong mastery).
      - confidence_level: "low", "medium", or "high".
      - affective_tone: short phrase describing emotional tone.
      - evidence_strength: "none", "weak", "medium", or "strong".

# Rules

- event_ids MUST be sequential with no gaps, starting at evt_0001.
- NOISE events MUST NOT contain an "analysis" key.
- ACADEMIC and AFFECT events MUST contain a complete "analysis" object.
- Use the human-readable speaker labels from speaker_map, never "SPEAKER_XX".
- Output must be valid, parseable JSON and nothing else.
"""


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load and parse a JSON file, with a clear error on failure."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input file not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc


def build_messages(
    transcript: Any, example_input: Any, example_output: Any
) -> list[dict[str, str]]:
    """Build the few-shot message list: example turn, then the real request."""
    example_output_json = json.dumps(example_output, ensure_ascii=False, indent=2)
    return [
        {
            "role": "user",
            "content": (
                "Convert this diarized transcript into the events JSON.\n\n"
                "<transcript>\n"
                + json.dumps(example_input, ensure_ascii=False, indent=2)
                + "\n</transcript>"
            ),
        },
        {"role": "assistant", "content": example_output_json},
        {
            "role": "user",
            "content": (
                "Convert this diarized transcript into the events JSON.\n\n"
                "<transcript>\n"
                + json.dumps(transcript, ensure_ascii=False, indent=2)
                + "\n</transcript>"
            ),
        },
    ]


def extract_json(text: str) -> str:
    """Strip markdown fences / surrounding prose and return the JSON substring."""
    stripped = text.strip()
    # Remove a leading ```json / ``` fence and trailing ``` if present.
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Otherwise, fall back to the outermost {...} span.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def call_model(client: anthropic.Anthropic, messages: list[dict[str, str]]) -> str:
    """Call the Anthropic API and return the model's raw text output."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    parts = [block.text for block in response.content if block.type == "text"]
    return "".join(parts)


def validate(result: Any) -> None:
    """Validate the parsed result against the event schema.

    Raises ValueError with a clear message on the first violation found.
    """
    if not isinstance(result, dict) or "events" not in result:
        raise ValueError("Top-level object must contain an 'events' key.")
    events = result["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("'events' must be a non-empty array.")

    ids: list[str] = []
    for i, event in enumerate(events):
        where = f"events[{i}]"
        if not isinstance(event, dict):
            raise ValueError(f"{where} must be an object.")

        required = [
            "event_id",
            "timestamp_range",
            "content_class",
            "event_type",
            "primary_speaker",
            "responders",
            "case_entities",
            "linked_event_id",
            "summary",
        ]
        missing = [k for k in required if k not in event]
        if missing:
            raise ValueError(f"{where} is missing required field(s): {missing}")

        eid = event["event_id"]
        if not isinstance(eid, str) or not EVENT_ID_RE.match(eid):
            raise ValueError(f"{where}.event_id {eid!r} must match 'evt_NNNN'.")
        expected = f"evt_{i + 1:04d}"
        if eid != expected:
            raise ValueError(
                f"{where}.event_id is {eid!r}; expected sequential {expected!r}."
            )
        ids.append(eid)

        if not TIMESTAMP_RANGE_RE.match(str(event["timestamp_range"])):
            raise ValueError(
                f"{where}.timestamp_range {event['timestamp_range']!r} "
                "must be 'HH:MM:SS - HH:MM:SS'."
            )

        if event["content_class"] not in CONTENT_CLASSES:
            raise ValueError(
                f"{where}.content_class {event['content_class']!r} "
                f"must be one of {sorted(CONTENT_CLASSES)}."
            )
        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(
                f"{where}.event_type {event['event_type']!r} "
                f"must be one of {sorted(EVENT_TYPES)}."
            )

        if not isinstance(event["primary_speaker"], str) or not event["primary_speaker"]:
            raise ValueError(f"{where}.primary_speaker must be a non-empty string.")
        if not isinstance(event["responders"], list):
            raise ValueError(f"{where}.responders must be an array.")

        ce = event["case_entities"]
        if not isinstance(ce, dict):
            raise ValueError(f"{where}.case_entities must be an object.")
        for key in ("patients", "eponyms_and_medical_terms"):
            if key not in ce or not isinstance(ce[key], list):
                raise ValueError(f"{where}.case_entities.{key} must be an array.")

        link = event["linked_event_id"]
        if link is not None and (not isinstance(link, str) or not EVENT_ID_RE.match(link)):
            raise ValueError(
                f"{where}.linked_event_id {link!r} must be null or 'evt_NNNN'."
            )

        if not isinstance(event["summary"], str) or not event["summary"]:
            raise ValueError(f"{where}.summary must be a non-empty string.")

        # analysis: required unless NOISE, forbidden for NOISE.
        if event["content_class"] == "NOISE":
            if "analysis" in event:
                raise ValueError(f"{where} is NOISE and must omit 'analysis'.")
        else:
            if "analysis" not in event:
                raise ValueError(
                    f"{where} is {event['content_class']} and must include 'analysis'."
                )
            analysis = event["analysis"]
            if not isinstance(analysis, dict):
                raise ValueError(f"{where}.analysis must be an object.")
            missing_a = [k for k in ANALYSIS_FIELDS if k not in analysis]
            if missing_a:
                raise ValueError(f"{where}.analysis is missing field(s): {missing_a}")
            if not isinstance(analysis["understanding_score"], int) or isinstance(
                analysis["understanding_score"], bool
            ):
                raise ValueError(f"{where}.analysis.understanding_score must be an integer.")
            if not 0 <= analysis["understanding_score"] <= 5:
                raise ValueError(f"{where}.analysis.understanding_score must be 0-5.")

    # linked_event_id must point to an existing event.
    id_set = set(ids)
    for i, event in enumerate(events):
        link = event["linked_event_id"]
        if link is not None and link not in id_set:
            raise ValueError(
                f"events[{i}].linked_event_id {link!r} does not match any event_id."
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a diarized transcript JSON into a conversational-events JSON."
    )
    # Find the first json file in the deidentified folder dynamically
    transcripts_dir = Path("data/deidentified_transcripts")
    try:
        first_json = next(transcripts_dir.glob("*.json"))
        default_input = str(first_json)
    except StopIteration:
        default_input = ""

    parser.add_argument(
        "input", 
        nargs="?", 
        default=default_input,
        help="Path to the input diarized transcript JSON."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/event_transcripts/test_output_events.json",
        help="Path to write the resulting events JSON.",
    )
    parser.add_argument(
        "--example-input",
        default=str(DEFAULT_EXAMPLE_INPUT),
        help="Few-shot example input transcript (default: bundled session1.json).",
    )
    parser.add_argument(
        "--example-output",
        default=str(DEFAULT_EXAMPLE_OUTPUT),
        help="Few-shot example gold output (default: bundled session12.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        return 2

    transcript = load_json(args.input)
    #example_input = load_json(args.example_input) Disabled reference script for connecting to deidentify
    #example_output = load_json(args.example_output)

    client = anthropic.Anthropic(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": f"Convert this diarized transcript into the events JSON.\n\n<transcript>\n{json.dumps(transcript, ensure_ascii=False, indent=2)}\n</transcript>"
        }
    ]

    print(f"Calling {MODEL} ...", file=sys.stderr)
    raw = call_model(client, messages)

    json_text = extract_json(raw)
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print("Error: model did not return valid JSON.", file=sys.stderr)
        print(f"Parse error: {exc}", file=sys.stderr)
        print("--- raw output (truncated) ---", file=sys.stderr)
        print(raw[:2000], file=sys.stderr)
        return 1

    validate(result)  # raises ValueError with a clear message on schema violation

    # 🔄 Dynamically determine the output filename based on the input filename
    input_path = Path(args.input)
    base_name = input_path.stem.replace("_deidentified", "") + "_events.json"
    out_path = Path("data/event_transcripts") / base_name
    
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(result['events'])} events to {out_path}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
