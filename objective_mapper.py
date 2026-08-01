#!/usr/bin/env python3
"""
objective_mapper.py
===================================================================================
Map structured medical-classroom conversational "events" to a canonical USMLE
curriculum CSV using ONE Anthropic LLM classification per event, then append a
"curriculum_mapping" block to each event while preserving everything else exactly.

GOAL (one sentence)
-------------------
For each event, use the Anthropic API to identify the SINGLE clinical concept the
conversation is MOST CENTRALLY about, map it to exactly one row of the curriculum
CSV, and append a "curriculum_mapping" block -- leaving every other field unchanged.

PIPELINE PER EVENT
------------------
    1. Local non-clinical short-circuit  -> no terms & no summary => null, no API call
    2. Local candidate prefilter         -> substring + rapidfuzz + optional aliases
                                            => bounded shortlist (+ a "none" sentinel)
    3. Persistent cache lookup           -> hit => reuse, skip API call
    4. ONE LLM call (temperature 0)      -> picks one shortlist topic_id (or "none")
                                            + one dimension; rules live in the system prompt
    5. Local reconcile + resolve         -> exact-first, else fuzzy >= threshold; the
                                            composite_numeric_id and subject are ALWAYS
                                            read from the CSV, never from the model
    6. Append curriculum_mapping LAST    -> four ordered keys, or all-null if unresolved

WHY THE CACHE, NOT temperature=0, IS THE REPRODUCIBILITY GUARANTEE
------------------------------------------------------------------
temperature=0 is near-deterministic but NOT guaranteed identical across calls/model
revisions. The persistent JSON cache is what makes repeated runs cheap and stable:
a cache hit reuses the prior decision and skips the API entirely.

NOTE ON --model DEFAULT
-----------------------
The task's suggested default `claude-3-5-haiku-20241022` reached end-of-life and now
returns HTTP 404. This script therefore defaults --model to `claude-haiku-4-5-20251001`
(a current Haiku) so it works out of the box. Override with --model as needed.

NOTE ON REGISTRY COLUMN NAMES
-----------------------------
Columns are indexed BY HEADER NAME (order-independent). The human-readable topic
column is accepted as either `topic_name` or `topic` (real MASTER_USMLE_TAXONOMY.csv
uses `topic`). Only the columns the pipeline actually consumes are required:
composite_numeric_id, subject, topic_id, (topic_name|topic), dimension.

EXAMPLE INVOCATION
------------------
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 objective_mapper.py \
        --registry "MASTER_USMLE_TAXONOMY.csv" \
        --input    "sample_input_objective_mapping.json" \
        --output   "enriched_output.json" \
        --aliases  "aliases.json" \
        --expected "sample_output_objective_mapping.json" \
        --cache    ".curriculum_cache.json" \
        --model    "claude-haiku-4-5-20251001"

No event data and no CSV rows are baked into this script; it processes ANY input of
the documented shape.
===================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import csv
import time
import random
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Guarded dependency imports (actionable messages instead of raw tracebacks)
# --------------------------------------------------------------------------- #
try:
    import anthropic  # REQUIRED: the LLM classifier
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "FATAL: the 'anthropic' package is required. Install it with:\n"
        "    pip install anthropic\n"
    )
    sys.exit(2)

try:
    from rapidfuzz import fuzz, process  # OPTIONAL: better prefilter + reconciliation
    _HAVE_RAPIDFUZZ = True
except ImportError:  # degrade to substring-only, do not crash
    _HAVE_RAPIDFUZZ = False
    sys.stderr.write(
        "WARNING: 'rapidfuzz' not installed -- falling back to substring-only "
        "candidate gathering and exact-only topic reconciliation.\n"
        "         For better recall install it with:  pip install rapidfuzz\n"
    )


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    """All human-facing progress goes to stderr; stdout/output file stays clean."""
    sys.stderr.write(msg + "\n")


def normalize(text: Any) -> str:
    """Canonical form for matching/hashing: casefold + collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def atomic_write_text(path: str, text: str) -> None:
    """Write text atomically (temp file in same dir + os.replace) so a crash never
    leaves a truncated file. os.replace is atomic on the same filesystem."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Clean up the temp file on any failure, then re-raise.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Registry (CSV) loading, validation, and indexing
# --------------------------------------------------------------------------- #

# Logical field -> accepted header names (case-insensitive). Only fields the
# pipeline actually consumes are required.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "composite_numeric_id": ["composite_numeric_id"],
    "subject": ["subject"],
    "topic_id": ["topic_id"],
    "topic_name": ["topic_name", "topic"],  # real CSV uses "topic"
    "dimension": ["dimension"],
}

# The three canonical self-test topics that must exist (with a "diagnosis"
# dimension) for the provided sample to resolve. Precheck only -- never hardcoded
# into any mapping.
SELFTEST_TOPICS = [
    "acute_coronary_syndrome",
    "acute_myocardial_infarction",
    "congestive_heart_failure",
]

NONE_SENTINEL = "none"  # LLM returns this topic_id when nothing in the shortlist fits


class Registry:
    """In-memory index over the taxonomy CSV. Loaded ONCE."""

    def __init__(self) -> None:
        self.topic_index: Dict[str, Dict[str, Dict[str, str]]] = {}  # topic_id -> {dimension -> row}
        self.subject_of: Dict[str, str] = {}                          # topic_id -> subject
        self.name_of: Dict[str, str] = {}                             # topic_id -> topic_name
        self.name_to_id: Dict[str, str] = {}                          # normalized topic_name -> topic_id
        self.valid_topic_ids: List[str] = []                          # sorted
        self.valid_dimensions: set = set()

    @classmethod
    def load(cls, path: str) -> "Registry":
        """Load + validate the CSV, failing fast with a clear message on any defect."""
        # encoding="utf-8-sig" transparently strips a BOM if present.
        try:
            fh = open(path, newline="", encoding="utf-8-sig")
        except OSError as exc:
            log(f"FATAL: cannot read registry '{path}': {exc}")
            sys.exit(2)

        with fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                log(f"FATAL: registry '{path}' has no header row.")
                sys.exit(2)

            header = [h.strip() for h in header]
            lower_to_actual = {h.casefold(): h for h in header}

            # Resolve each required logical field to an actual header name.
            resolved: Dict[str, str] = {}
            missing: List[str] = []
            for logical, candidates in COLUMN_ALIASES.items():
                match = next((lower_to_actual[c.casefold()] for c in candidates
                              if c.casefold() in lower_to_actual), None)
                if match is None:
                    missing.append(f"{logical} (accepted: {', '.join(candidates)})")
                else:
                    resolved[logical] = match
            if missing:
                log("FATAL: registry is missing required column(s):\n  - "
                    + "\n  - ".join(missing)
                    + f"\n  Found columns: {header}")
                sys.exit(2)

            col_idx = {logical: header.index(actual) for logical, actual in resolved.items()}

            reg = cls()
            data_rows = 0
            for raw in reader:
                if not raw or all(str(c).strip() == "" for c in raw):
                    continue  # skip blank lines
                data_rows += 1

                def cell(logical: str) -> str:
                    i = col_idx[logical]
                    return str(raw[i]).strip() if i < len(raw) else ""

                topic_id = cell("topic_id")
                dimension = cell("dimension")
                if not topic_id or not dimension:
                    continue

                row = {
                    "composite_numeric_id": cell("composite_numeric_id"),
                    "subject": cell("subject"),
                    "topic_id": topic_id,
                    "topic_name": cell("topic_name"),
                    "dimension": dimension,
                }

                dim_map = reg.topic_index.setdefault(topic_id, {})
                if dimension in dim_map:
                    # Duplicate (topic_id, dimension): warn, keep first.
                    log(f"WARNING: duplicate (topic_id, dimension)="
                        f"({topic_id}, {dimension}); keeping first, ignoring later row.")
                    continue
                dim_map[dimension] = row
                reg.subject_of.setdefault(topic_id, row["subject"])
                reg.name_of.setdefault(topic_id, row["topic_name"])
                reg.valid_dimensions.add(dimension)

        if data_rows == 0:
            log(f"FATAL: registry '{path}' has zero data rows.")
            sys.exit(2)

        # Build the normalized-name -> topic_id index (first writer wins).
        for tid, name in reg.name_of.items():
            reg.name_to_id.setdefault(normalize(name), tid)
        reg.valid_topic_ids = sorted(reg.topic_index.keys())

        log(f"Loaded registry: {len(reg.valid_topic_ids)} topics, "
            f"{data_rows} rows, {len(reg.valid_dimensions)} dimensions.")
        return reg

    def completeness_precheck(self) -> None:
        """Warn (do not fail) if any canonical self-test topic lacks a diagnosis row,
        so a resulting null is diagnosed as a DATA problem, not a code/LLM problem."""
        for tid in SELFTEST_TOPICS:
            if "diagnosis" not in self.topic_index.get(tid, {}):
                log("WARNING: registry INCOMPLETE -- expected topic "
                    f"'{tid}' with a 'diagnosis' dimension is missing. "
                    "Any event centered on it CANNOT resolve (data problem).")


# --------------------------------------------------------------------------- #
# Candidate prefilter (local, BEFORE the LLM call)
# --------------------------------------------------------------------------- #
#
# KNOWN LIMITATION (by design): the prefilter ranks by STRING similarity only and
# is blind to meaning. It WILL include high-scoring distractors driven by secondary
# terms -- e.g. "hypertension" pulling a hypertension topic to score 100 on an event
# that actually concludes heart failure. That is acceptable: the prefilter only
# NARROWS the space; the LLM makes the semantic decision. The prefilter must NEVER
# auto-pick the top-scored candidate.

SUBSTRING_SCORE = 100.0  # fixed high score for a containment hit


def _probe_score(probe: str, topic_name: str) -> float:
    """Best score of one probe string against one topic_name."""
    p, n = probe.casefold().strip(), topic_name.casefold().strip()
    if not p or not n:
        return 0.0
    if p in n or n in p:  # (a) case-insensitive substring containment
        return SUBSTRING_SCORE
    if _HAVE_RAPIDFUZZ:    # (b) fuzzy token overlap
        return float(fuzz.token_set_ratio(p, n))
    return 0.0


def build_shortlist(
    terms: List[str],
    summary: str,
    reg: Registry,
    aliases: Dict[str, List[str]],
    shortlist_size: int,
    active_topics: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Gathers candidates using weighted term scoring and exact term matching."""
    probes = [str(t).strip() for t in terms if str(t).strip()]
    best: Dict[str, float] = {}

    # 1. Primary Matching: Score taxonomy topics against extracted medical terms
    for tid in reg.valid_topic_ids:
        name = reg.name_of.get(tid, tid)
        tid_clean = tid.replace("_", " ")
        score = 0.0

        for probe in probes:
            p_lower, n_lower, t_lower = probe.lower(), name.lower(), tid_clean.lower()

            # Boost exact word or substring overlap (e.g., 'appendicitis' -> 'acute_appendicitis')
            if p_lower in n_lower or p_lower in t_lower or n_lower in p_lower:
                s = 90.0
            elif _HAVE_RAPIDFUZZ:
                s = max(fuzz.WRatio(p_lower, n_lower), fuzz.WRatio(p_lower, t_lower))
            else:
                s = 50.0 if (p_lower in n_lower or p_lower in t_lower) else 0.0

            if s > score:
                score = float(s)

        best[tid] = score

    # 2. Alias expansion
    for tid, alias_list in aliases.items():
        if tid not in reg.topic_index:
            continue
        for alias in alias_list:
            for probe in probes:
                s = _probe_score(alias, probe)
                if s > best.get(tid, 0.0):
                    best[tid] = s

    # 3. Fallback to summary ONLY if medical terms yielded zero solid matches
    if max(best.values(), default=0) < 40 and str(summary).strip():
        for tid in reg.valid_topic_ids:
            name = reg.name_of.get(tid, tid)
            if _HAVE_RAPIDFUZZ:
                s = float(fuzz.partial_ratio(name.lower(), summary.lower())) * 0.5
            else:
                s = 40.0 if name.lower() in summary.lower() else 0.0
            if s > best.get(tid, 0.0):
                best[tid] = s
    # 3.5 PIN ACTIVE CASE TOPICS (Carries forward context from previous events)
    if active_topics:
        for tid in active_topics:
            if tid in reg.topic_index:
                best[tid] = 100.0  # <--- FORCE ACTIVE TOPICS TO TOP OF SHORTLIST

    # 4. Rank candidates by score descending
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    
    # Guarantee at least 40 shortlist candidates reach Claude to prevent false nulls
    effective_size = max(shortlist_size, 40)
    top = [tid for tid, score in ranked[:effective_size] if score > 0]

    if not top:
        top = [tid for tid, _ in ranked[:effective_size]]

    shortlist = [
        {
            "topic_id": tid,
            "topic_name": reg.name_of.get(tid, tid),
            "available_dimensions": sorted(reg.topic_index[tid].keys()),
        }
        for tid in top
    ]
    shortlist.append({
        "topic_id": NONE_SENTINEL,
        "topic_name": "None of these fit / not a clinical concept",
        "available_dimensions": [],
    })
    return shortlist


# --------------------------------------------------------------------------- #
# Persistent cache
# --------------------------------------------------------------------------- #

def cache_key(terms: List[str], summary: str, model: str, cache_version: str) -> str:
    """Stable hash of EXACTLY (sorted terms, normalized summary, model, version).
    The shortlist is deliberately NOT part of the key: it changes as the registry
    grows even when the correct answer would not, which would cause confusing
    invalidation. Bump --cache-version manually to intentionally discard answers."""
    payload = json.dumps(
        {
            "terms": sorted(str(t) for t in terms),
            "summary": normalize(summary),
            "model": model,
            "version": cache_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: could not read cache '{path}' ({exc}); starting empty.")
        return {}


def save_cache(path: Optional[str], cache: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        atomic_write_text(path, json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        log(f"WARNING: could not write cache '{path}': {exc}")


# --------------------------------------------------------------------------- #
# LLM classification (one call per event with content, temperature 0)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You classify a medical-classroom discussion to the SINGLE clinical concept it is \
MOST CENTRALLY about, choosing from a provided candidate shortlist.

You are given only two things about the discussion: a list of medical terms and a \
short summary. Apply ALL of these rules:

CLINICAL FOCUS rule: ALWAYS prioritize core clinical conditions, diseases, pathology, \
or symptoms (e.g., appendicitis, diverticulitis) over cognitive, educational, or \
behavioral meta-concepts (e.g., anchoring bias, communication errors) unless the \
discussion is explicitly a social sciences or ethics lecture.

PRIMARY-CONCEPT rule (CRITICAL): a term merely APPEARING in the term list does NOT \
make it the answer. Terms are often secondary context, comparisons, or things \
explicitly ruled out, and such distractors frequently score high in the shortlist. \
Map to the concept the DIALOGUE CENTERS ON and CONCLUDES ON -- not the most-mentioned \
or first-listed term.
  Example: an event listing "ACS" and "acute MI" whose summary rules ACS out and \
whose diagnostic reasoning concludes on myocardial infarction maps to \
acute_myocardial_infarction, NOT acute_coronary_syndrome.
  Example: an event listing "hypertension" only as background but concluding heart \
failure maps to congestive_heart_failure, NOT any hypertension topic -- even though a \
hypertension topic may score 100 in the shortlist.

LATENT-TOPIC rule: the correct topic may NOT appear literally in the terms. \
"heart failure" / "systolic heart failure" -> congestive_heart_failure.

RULED-OUT rule: if a concept is mentioned only as SECONDARY background or comparison \
and the summary excludes it ("not X", "isn't X", "rules it out", "more likely than X", \
"secondary marker"), it is NOT the answer even if it scored highly or is listed first. \
IMPORTANT DISTINCTION: this applies to secondary/background concepts. If the ENTIRE \
discussion is centered on arguing whether ONE concept is the diagnosis -- even if it \
concludes that concept is ruled out as the main diagnosis -- that concept is what the \
event is ABOUT, so map to it.

TIE-BREAK rule: if two or more topics remain plausible, pick the one MOST \
REPRESENTATIVE of the conversation as a whole -- what it centers on and concludes with.

DIMENSION rule: choose the dimension matching HOW the concept is discussed. DEFAULT to \
"diagnosis" when the event is about identifying/arguing/ruling a diagnosis in or out. \
Choose management / prognosis / pathophysiology / presentation only when the summary is \
clearly about that facet.

Pick EXACTLY ONE topic_id from the shortlist (or "none" if truly nothing fits), and \
ONE dimension from that topic's available_dimensions. Return ONLY a raw JSON object:
{"topic_id": "...", "dimension": "..."}
No prose, no markdown, no code fences. topic_id may be "none".\
"""


def build_user_message(terms: List[str], summary: str, shortlist: List[Dict[str, Any]]) -> str:
    """Compact per-event message: the two scope fields + the enumerated shortlist."""
    lines = [
        "SUMMARY:",
        summary.strip() or "(empty)",
        "",
        "MEDICAL TERMS:",
        json.dumps(list(terms), ensure_ascii=False),
        "",
        "CANDIDATE SHORTLIST (choose exactly one topic_id, or \"none\"):",
    ]
    for i, c in enumerate(shortlist, 1):
        dims = ", ".join(c["available_dimensions"]) if c["available_dimensions"] else "-"
        lines.append(f'{i}. topic_id="{c["topic_id"]}" | name="{c["topic_name"]}" '
                     f"| available_dimensions=[{dims}]")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Strip ``` / ```json fences and parse the first JSON object found."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def classify_with_llm(
    client: "anthropic.Anthropic",
    model: str,
    terms: List[str],
    summary: str,
    shortlist: List[Dict[str, Any]],
    max_attempts: int = 3,
) -> Optional[Dict[str, str]]:
    """One classification call with retry+backoff on 429 / >=500. Returns
    {"topic_id","dimension"} or None on unrecoverable failure (caller emits null)."""
    user_msg = build_user_message(terms, summary, shortlist)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )
            parsed = _extract_json_object(raw)
            if parsed is None:
                log(f"    WARNING: unparseable model output (attempt {attempt}): {raw!r}")
                last_error = "unparseable"
                continue
            return {
                "topic_id": str(parsed.get("topic_id", "")).strip(),
                "dimension": str(parsed.get("dimension", "")).strip(),
            }
        except Exception as exc:  # noqa: BLE001 - classify transient vs fatal below
            status = getattr(exc, "status_code", None)
            transient = (
                status in (429, 500, 502, 503, 504, 529)
                or isinstance(exc, getattr(anthropic, "APIConnectionError", ()))
                or isinstance(exc, getattr(anthropic, "RateLimitError", ()))
                or isinstance(exc, getattr(anthropic, "InternalServerError", ()))
            )
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            if transient and attempt < max_attempts:
                # Exponential backoff with jitter.
                delay = min(8.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                log(f"    transient API error (attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay:.2f}s: {last_error}")
                time.sleep(delay)
                continue
            log(f"    API error (attempt {attempt}): {last_error}")
            break
    log(f"    giving up on this event after {max_attempts} attempt(s): {last_error}")
    return None


# --------------------------------------------------------------------------- #
# Reconcile + resolve (local, authoritative)
# --------------------------------------------------------------------------- #

NULL_MAPPING = {
    "composite_numeric_id": None,
    "subject": None,
    "topic_id": None,
    "dimension": None,
}


def reconcile_topic_id(model_topic: str, reg: Registry, fuzzy_threshold: int,
                       event_id: str) -> Optional[str]:
    """Exact-first, else fuzzy-reconcile a near-miss topic_id to a real one."""
    if not model_topic:
        return None
    if model_topic in reg.topic_index:  # EXACT-FIRST
        return model_topic

    if not _HAVE_RAPIDFUZZ:
        log(f"    [{event_id}] unresolved topic '{model_topic}' "
            "(rapidfuzz unavailable; exact-only).")
        return None

    probe = normalize(model_topic)
    # Reconcile against valid topic_ids AND against normalized topic_names.
    cand_ids = process.extractOne(probe, reg.valid_topic_ids,
                                  scorer=fuzz.token_sort_ratio)
    cand_names = process.extractOne(probe, list(reg.name_to_id.keys()),
                                    scorer=fuzz.token_sort_ratio)

    best_id, best_score = None, -1.0
    if cand_ids and cand_ids[1] > best_score:
        best_id, best_score = cand_ids[0], cand_ids[1]
    if cand_names and cand_names[1] > best_score:
        best_id, best_score = reg.name_to_id[cand_names[0]], cand_names[1]

    if best_id is not None and best_score >= fuzzy_threshold:
        log(f"    [{event_id}] reconciled topic '{model_topic}' -> "
            f"'{best_id}' (score {best_score:.0f} >= {fuzzy_threshold}).")
        return best_id

    log(f"    [{event_id}] could NOT reconcile topic '{model_topic}' "
        f"(best '{best_id}' score {best_score:.0f} < {fuzzy_threshold}).")
    return None


def resolve_mapping(model_result: Optional[Dict[str, str]], reg: Registry,
                    fuzzy_threshold: int, event_id: str) -> Dict[str, Any]:
    """Turn a model decision into the authoritative four-key mapping (or null).
    composite_numeric_id and subject ALWAYS come from the CSV, never the model."""
    if not model_result:
        return dict(NULL_MAPPING)

    model_topic = model_result.get("topic_id", "")
    model_dim = model_result.get("dimension", "")

    # "none" sentinel or empty -> a legitimate all-null outcome.
    if not model_topic or model_topic.strip().casefold() == NONE_SENTINEL:
        log(f"    [{event_id}] model chose 'none' -> null mapping.")
        return dict(NULL_MAPPING)

    topic_id = reconcile_topic_id(model_topic, reg, fuzzy_threshold, event_id)
    if topic_id is None:
        log(f"    WARNING [{event_id}]: Model returned '{model_topic}', "
        f"but it could not be matched to CSV taxonomy -> null mapping.")
        return dict(NULL_MAPPING)

    dim_map = reg.topic_index[topic_id]
    # Validate/repair the dimension: prefer the model's, else 'diagnosis', else first.
    if model_dim in dim_map:
        dimension = model_dim
    else:
        dimension = "diagnosis" if "diagnosis" in dim_map else sorted(dim_map.keys())[0]
        log(f"    [{event_id}] dimension '{model_dim}' invalid for '{topic_id}' "
            f"-> substituted '{dimension}'.")

    row = dim_map[dimension]
    return {
        "composite_numeric_id": row["composite_numeric_id"],  # from CSV
        "subject": row["subject"],                            # from CSV
        "topic_id": topic_id,
        "dimension": dimension,
    }


# --------------------------------------------------------------------------- #
# Per-event orchestration
# --------------------------------------------------------------------------- #

def process_event(
    event: Dict[str, Any],
    reg: Registry,
    aliases: Dict[str, List[str]],
    client: Optional["anthropic.Anthropic"],
    cache: Dict[str, Any],
    args: argparse.Namespace,
    active_topics: Optional[set] = None,  # <--- 1. ADD THIS PARAMETER
) -> Dict[str, Any]:
    """Run the full pipeline for one event and return its curriculum_mapping dict."""
    event_id = str(event.get("event_id", "<no-id>"))
    case_entities = event.get("case_entities", {}) or {}
    terms = case_entities.get("eponyms_and_medical_terms", []) or []
    if not isinstance(terms, list):
        terms = []
    summary = event.get("summary", "") or ""

    # STAGE 1 -- local non-clinical short-circuit (no API call).
    if not terms and not str(summary).strip():
        log(f"[{event_id}] non_clinical (local skip) -> null mapping.")
        return dict(NULL_MAPPING)

    # STAGE 3 -- cache lookup (keyed independently of the shortlist).
    key = cache_key(terms, summary, args.model, args.cache_version)
    if key in cache:
        cached = cache[key]
        log(f"[{event_id}] cache HIT -> topic_id={cached.get('topic_id')!r} "
            f"dimension={cached.get('dimension')!r}")
        mapping = resolve_mapping(cached, reg, args.fuzzy_threshold, event_id)
        return mapping

    # STAGE 2 -- local candidate prefilter (narrows; never decides).
    shortlist = build_shortlist(
        terms, summary, reg, aliases, args.shortlist_size, active_topics=active_topics  # <--- 2. PASS IT HERE
    )

    # STAGE 4 -- one LLM classification call.
    if client is None:
        log(f"[{event_id}] no API client -> null mapping.")
        return dict(NULL_MAPPING)
    log(f"[{event_id}] cache MISS -> llm-call ({len(shortlist)-1} candidates + sentinel)")
    model_result = classify_with_llm(
        client, args.model, terms, summary, shortlist
    )
    print(f"DEBUG [{event_id}] Claude RAW returned: {model_result}")

    # STAGE 5 -- reconcile + resolve locally (authoritative).
    mapping = resolve_mapping(model_result, reg, args.fuzzy_threshold, event_id)

    # Cache the raw model decision (topic_id + dimension), not the resolved row, so a
    # later registry fix can re-resolve the same decision without a new API call.
    if model_result is not None:
        cache[key] = {"topic_id": model_result.get("topic_id", ""),
                      "dimension": model_result.get("dimension", "")}

    log(f"[{event_id}] resolved -> topic_id={mapping['topic_id']!r} "
        f"dimension={mapping['dimension']!r} id={mapping['composite_numeric_id']!r}")
    return mapping


def enrich_event(event: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the event and append curriculum_mapping as the LAST key, explicitly, so
    original key order is preserved on every Python version. No other field changes."""
    new_event = {k: v for k, v in event.items() if k != "curriculum_mapping"}
    new_event["curriculum_mapping"] = {
        "composite_numeric_id": mapping["composite_numeric_id"],
        "subject": mapping["subject"],
        "topic_id": mapping["topic_id"],
        "dimension": mapping["dimension"],
    }
    return new_event


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def run_self_test(produced: Dict[str, Any], expected_path: str) -> bool:
    """Compare produced curriculum_mapping to an expected file; print PASS/FAIL."""
    try:
        with open(expected_path, encoding="utf-8") as fh:
            expected = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"SELF-TEST ERROR: cannot read expected '{expected_path}': {exc}")
        return False

    exp_by_id = {str(e.get("event_id")): (e.get("curriculum_mapping") or {})
                 for e in expected.get("events", [])}

    log("\n===== SELF-TEST (expected vs got) =====")
    header = f"{'event_id':<12} {'result':<6} {'field':<10} {'expected':<32} got"
    log(header)
    log("-" * len(header))
    all_pass = True
    for e in produced.get("events", []):
        eid = str(e.get("event_id"))
        got = e.get("curriculum_mapping") or {}
        exp = exp_by_id.get(eid)
        if exp is None:
            log(f"{eid:<12} {'SKIP':<6} (no expected entry)")
            continue
        for field in ("topic_id", "dimension", "composite_numeric_id"):
            ev, gv = exp.get(field), got.get(field)
            ok = ev == gv
            all_pass = all_pass and ok
            log(f"{eid:<12} {('PASS' if ok else 'FAIL'):<6} {field:<10} "
                f"{str(ev):<32} {gv}")
    log("=======================================")
    log("SELF-TEST: " + ("ALL PASS" if all_pass else "FAILURES PRESENT"))
    return all_pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Map medical-classroom events to a curriculum CSV via one LLM "
                    "classification per event.")
    p.add_argument("--registry", required=True, help="Path to master taxonomy CSV.")
    p.add_argument("--input", required=True, help="Path to input events JSON.")
    p.add_argument("--output", required=True, help="Path to write enriched JSON.")
    p.add_argument("--cache", default=".curriculum_cache.json",
                   help="Path to JSON LLM cache (default: .curriculum_cache.json).")
    p.add_argument("--cache-version", default="v1",
                   help="Bump to force cache invalidation (default: v1).")
    p.add_argument("--aliases", default=None,
                   help="Optional JSON map: topic_id -> [alias strings].")
    p.add_argument("--expected", default=None,
                   help="Optional expected-output JSON for self-test.")
    # NOTE: spec default 'claude-3-5-haiku-20241022' is EOL/404; using a current Haiku.
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Anthropic model id (default: claude-haiku-4-5-20251001).")
    p.add_argument("--shortlist-size", type=int, default=25,
                   help="Max candidates sent to the LLM (default: 25).")
    p.add_argument("--fuzzy-threshold", type=int, default=80,
                   help="0-100 threshold to reconcile near-miss topic_ids (default: 80).")
    return p.parse_args(argv)


def load_aliases(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: could not read aliases '{path}' ({exc}); ignoring.")
        return {}
    out: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for tid, al in data.items():
            if isinstance(al, list):
                out[str(tid)] = [str(a) for a in al]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # API key must be present BEFORE any processing. Never print its value.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("FATAL: ANTHROPIC_API_KEY is not set. Export it and retry.")
        return 2

    # Load registry (fails fast on defects) + completeness precheck.
    reg = Registry.load(args.registry)
    reg.completeness_precheck()

    aliases = load_aliases(args.aliases)
    if aliases:
        log(f"Loaded {len(aliases)} alias group(s).")

    # Load input events.
    try:
        with open(args.input, encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log(f"FATAL: cannot read input '{args.input}': {exc}")
        return 2
    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        log("FATAL: input must be a JSON object with an 'events' array.")
        return 2

    cache = load_cache(args.cache)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

   # Process every event, in order. One bad event never aborts the run.
    enriched_events: List[Dict[str, Any]] = []
    active_topics: set = set()

    for event in document["events"]:
        if not isinstance(event, dict):
            enriched_events.append(event)  # pass through anything unexpected untouched
            continue
        try:
            mapping = process_event(
                event, reg, aliases, client, cache, args, active_topics=active_topics
            )
            resolved_topic = mapping.get("topic_id")
            if resolved_topic and resolved_topic != NONE_SENTINEL:
                active_topics.add(resolved_topic)
        except Exception as exc:  # noqa: BLE001 - never let one event kill the run
            traceback.print_exc()
            log(f"WARNING [{event.get('event_id','<no-id>')}]: unexpected error "
                f"({type(exc).__name__}: {exc}); emitting null mapping.")
            mapping = dict(NULL_MAPPING)
        enriched_events.append(enrich_event(event, mapping))

    # Persist the cache (best-effort, atomic).
    save_cache(args.cache, cache)

    # Assemble output: session_metadata FIRST (only if present), then events.
    out_doc: Dict[str, Any] = {}
    if "session_metadata" in document:
        out_doc["session_metadata"] = document["session_metadata"]
    out_doc["events"] = enriched_events

    # Serialize: indent=2, literal Unicode, default separators, one trailing newline,
    # atomic write so a crash never leaves a truncated output.
    text = json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n"
    try:
        atomic_write_text(args.output, text)
    except OSError as exc:
        log(f"FATAL: could not write output '{args.output}': {exc}")
        return 2

    # Verify the file round-trips.
    with open(args.output, encoding="utf-8") as fh:
        assert json.load(fh) == out_doc, "output failed round-trip check"
    log(f"Wrote {len(enriched_events)} event(s) -> {args.output}")

    # Optional self-test; nonzero exit on any mismatch.
    if args.expected:
        if not run_self_test(out_doc, args.expected):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
