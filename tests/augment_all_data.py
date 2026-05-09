"""
augment_all_data.py

LLM-based data augmentation for tests/all_data.csv.

It keeps the original schema:
    sentence;performative;embedding;solution;domain

Only sentence and solution are augmented. performative/embedding/domain are kept
unchanged, as in the original row.

Improvements over v1:
  - Incremental write: augmented rows are appended to output file immediately,
    so a crash never loses already-generated data.
  - Retry with exponential backoff: each API call is retried up to MAX_RETRIES
    times on transient errors (network, rate-limit 429, etc.).
  - Deduplication: sentences that are identical (case-insensitive) to already
    seen sentences are silently dropped.
  - Arity validation: the generated solution's functor and argument count are
    compared against the embedding field (e.g. "check_in/2"). Variants that
    do not match are discarded with a warning.

Usage:
    set OPENAI_API_KEY=your_key_here
    python tests/augment_all_data.py
    python tests/augment_all_data.py --per-row 5 --model gpt-4o-mini --delay 0.5
    python tests/augment_all_data.py --start 0 --limit 50
"""

import argparse
import csv
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# ── Retry settings ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds; wait = BACKOFF_BASE ** attempt


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM augmentation for all_data.csv")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input", default=os.path.join(base_dir, "all_data.csv"), help="Input CSV path")
    parser.add_argument(
        "--output",
        default=os.path.join(base_dir, "all_data_augmented.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--per-row", type=int, default=3, help="How many variants to request per row")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between API calls")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resume/chunk runs)")
    parser.add_argument("--limit", type=int, default=0, help="How many rows to process from start (0 = all)")
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="If set, output contains only augmented rows",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional custom OpenAI-compatible base URL",
    )
    return parser.parse_args()


# ── OpenAI client ─────────────────────────────────────────────────────────────
def build_client(base_url: str) -> OpenAI:
    if base_url.strip():
        return OpenAI(base_url=base_url.strip())
    return OpenAI()


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(row: Dict[str, str], num_variants: int) -> str:
    sentence = row["sentence"]
    performative = row["performative"]
    embedding = row["embedding"]
    solution = row["solution"]
    domain = row["domain"]

    return f"""You are an expert in NLP data augmentation for NL-to-logic tasks.
Your goal is to produce natural sentence variants that a real user would type in a chatbot.

Given one training example, generate {num_variants} new variations.

Output format — return ONLY plain semicolon-separated lines, one per variant:
new_sentence;new_solution

Hard constraints:
1) Each line must be: new_sentence;new_solution — nothing else.
2) No headers, numbering, markdown, code fences, or extra text.
3) The new sentence must keep the same semantic intent as the original.
4) Update the logical solution to match the new sentence exactly.
5) Keep the same performative ({performative}), functor and arity as the original.
6) Use valid Prolog/Jason-like syntax in the solution.
7) Write in English. Do NOT repeat the original sentence.
8) The solution functor and number of arguments MUST match the embedding exactly: {embedding}.

STYLE — write as a real user typing in a chatbot (casual, direct, fluent):
- Natural phrasing: vary word order, vocabulary, sentence structure genuinely.
- FORBIDDEN prefixes: never start with "FYI", "Update:", "For the record,",
  "Reminder:", "Note:", "Heads up:", "Just so you know", "Please note",
  "Quick check:", "Attention:", or any similar formal/email-style opener.
- FORBIDDEN pattern: never prepend "Can you", "Could you", "Please" or any
  modal verb to an already-formed question without rewriting the whole sentence.
  BAD:  "Can you Are there any open doors?" — this is broken English.
  GOOD: "Which doors are still open?" — rewrite from scratch.
- Every sentence must be grammatically correct and self-contained.

Reference context (do NOT output these):
performative={performative}
embedding={embedding}
domain={domain}

Original row:
{sentence};{solution}

Generate {num_variants} lines:"""


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_llm_lines(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ";" not in line:
            continue
        left, right = line.split(";", 1)
        sentence = left.strip()
        solution = right.strip()
        if sentence and solution:
            pairs.append((sentence, solution))
    return pairs


# ── Arity validation ──────────────────────────────────────────────────────────
def parse_embedding(embedding: str) -> Optional[Tuple[str, int]]:
    """
    Parse an embedding like 'check_in/2' into ('check_in', 2).
    Returns None if the format is not recognisable.
    """
    m = re.fullmatch(r"([a-zA-Z_]\w*)/(\d+)", embedding.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def count_top_level_args(args_str: str) -> int:
    """
    Count top-level comma-separated arguments inside a functor's parentheses,
    handling nested parentheses correctly.
    E.g. 'gigi, room(12, deluxe)' → 2
    """
    depth = 0
    count = 1
    for ch in args_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def validate_solution(solution: str, embedding: str) -> bool:
    """
    Return True if *solution* has the same functor and arity as *embedding*.
    If the embedding format is unrecognised, allow the solution through.
    """
    expected = parse_embedding(embedding)
    if expected is None:
        return True  # can't check → pass through

    expected_functor, expected_arity = expected

    # Match: functor_name(...)
    m = re.fullmatch(r"([a-zA-Z_]\w*)\((.*)\)\s*", solution.strip(), re.DOTALL)
    if not m:
        # Accept 0-arity atoms when expected_arity == 0
        if expected_arity == 0 and solution.strip() == expected_functor:
            return True
        return False

    actual_functor = m.group(1)
    args_str = m.group(2).strip()

    if actual_functor != expected_functor:
        return False

    actual_arity = count_top_level_args(args_str) if args_str else 0
    return actual_arity == expected_arity


# ── Retry wrapper ─────────────────────────────────────────────────────────────
def request_variants(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    per_row: int,
    temperature: float,
) -> List[Tuple[str, str]]:
    """Call the API with automatic retry + exponential backoff."""
    prompt = build_prompt(row, per_row)
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only semicolon-separated lines: sentence;solution",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return parse_llm_lines(content)
        except Exception as exc:
            last_exc = exc
            wait = BACKOFF_BASE**attempt
            print(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {exc} — waiting {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed: {last_exc}") from last_exc


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = build_client(args.base_url)

    with open(args.input, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Input CSV has no header")

    required = ["sentence", "performative", "embedding", "solution", "domain"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    start = max(0, args.start)
    end = len(rows) if args.limit <= 0 else min(len(rows), start + args.limit)
    selected_rows = rows[start:end]

    # ── Incremental output setup ───────────────────────────────────────────────
    # If resuming (--start > 0) and the output file already exists, we append
    # without re-writing the header so progress is preserved across runs.
    out_file_exists = os.path.isfile(args.output)
    resuming = args.start > 0 and out_file_exists

    out_f = open(args.output, "a" if resuming else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter=";")

    if not resuming:
        writer.writeheader()

    # ── Seed the seen-sentences set from any rows already in the output ────────
    seen_sentences: set = set()

    if out_file_exists:
        with open(args.output, encoding="utf-8", newline="") as existing_f:
            for r in csv.DictReader(existing_f, delimiter=";"):
                seen_sentences.add(r["sentence"].strip().lower())

    # Write originals on a fresh run (not resuming)
    if not args.no_original and not resuming:
        for row in rows:
            key = row["sentence"].strip().lower()
            if key not in seen_sentences:
                writer.writerow(row)
                seen_sentences.add(key)
        out_f.flush()

    # ── Augmentation loop ──────────────────────────────────────────────────────
    total_aug = 0
    total_skipped_dup = 0
    total_skipped_arity = 0
    total_errors = 0

    for idx, row in enumerate(selected_rows, start=start):
        print(f"[{idx + 1}/{len(rows)}] Augmenting: {row['sentence']}")
        try:
            pairs = request_variants(
                client=client,
                model=args.model,
                row=row,
                per_row=args.per_row,
                temperature=args.temperature,
            )
            for sentence, solution in pairs:
                # 1) Deduplication
                key = sentence.strip().lower()
                if key in seen_sentences:
                    total_skipped_dup += 1
                    print(f"  [DUP]   skipped duplicate: {sentence}")
                    continue

                # 2) Arity validation
                if not validate_solution(solution, row["embedding"]):
                    total_skipped_arity += 1
                    print(
                        f"  [ARITY] skipped (embedding={row['embedding']}): {solution}"
                    )
                    continue

                # 3) Write immediately
                new_row = {
                    "sentence": sentence,
                    "performative": row["performative"],
                    "embedding": row["embedding"],
                    "solution": solution,
                    "domain": row["domain"],
                }
                writer.writerow(new_row)
                out_f.flush()
                seen_sentences.add(key)
                total_aug += 1

        except Exception as exc:
            total_errors += 1
            print(f"  [WARN] row {idx + 1} failed: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)

    out_f.close()

    print("\nDone")
    print(f"Input rows            : {len(rows)}")
    print(f"Processed rows        : {len(selected_rows)}")
    print(f"Augmented rows added  : {total_aug}")
    print(f"Skipped (duplicate)   : {total_skipped_dup}")
    print(f"Skipped (arity error) : {total_skipped_arity}")
    print(f"Rows failed           : {total_errors}")
    print(f"Output                : {args.output}")


if __name__ == "__main__":
    main()
