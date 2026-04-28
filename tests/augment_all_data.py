"""
augment_all_data.py

LLM-based data augmentation for tests/all_data.csv.

It keeps the original schema:
    sentence;performative;embedding;solution;domain

Only sentence and solution are augmented. performative/embedding/domain are kept
unchanged, as in the original row.

Usage:
    set OPENAI_API_KEY=your_key_here
    python tests/augment_all_data.py
    python tests/augment_all_data.py --per-row 3 --model gpt-4o-mini
    python tests/augment_all_data.py --start 0 --limit 50
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

from openai import OpenAI


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
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between API calls")
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


def build_client(base_url: str) -> OpenAI:
    if base_url.strip():
        return OpenAI(base_url=base_url.strip())
    return OpenAI()


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


def request_variants(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    per_row: int,
    temperature: float,
) -> List[Tuple[str, str]]:
    prompt = build_prompt(row, per_row)
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

    out_rows: List[Dict[str, str]] = []
    if not args.no_original:
        out_rows.extend(rows)

    total_aug = 0
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
                out_rows.append(
                    {
                        "sentence": sentence,
                        "performative": row["performative"],
                        "embedding": row["embedding"],
                        "solution": solution,
                        "domain": row["domain"],
                    }
                )
                total_aug += 1
        except Exception as exc:
            total_errors += 1
            print(f"  [WARN] row {idx + 1} failed: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nDone")
    print(f"Input rows            : {len(rows)}")
    print(f"Processed rows        : {len(selected_rows)}")
    print(f"Augmented rows added  : {total_aug}")
    print(f"Rows failed           : {total_errors}")
    print(f"Total output rows     : {len(out_rows)}")
    print(f"Output                : {args.output}")


if __name__ == "__main__":
    main()
