"""
augment_all_data.py  —  Professional LLM-based data augmentation

Generates paraphrased training rows for NL-to-Prolog datasets.
Each original (sentence, solution) pair is sent to an LLM that produces
N new natural-language variants with correctly updated Prolog solutions.

Key features:
  - Literals-aware prompt: feeds domain examples from all_literals.csv so
    the LLM understands quoting, compound terms, and argument types.
  - Structural validation: checks functor, arity, and per-position argument
    type (number, quoted, atom, compound) against all_literals.csv.
  - Incremental write: rows are flushed to disk immediately — a crash never
    loses already-generated data. Resume with --start N.
  - Retry with exponential backoff on transient API errors.
  - Deduplication: case-insensitive sentence dedup.

Usage:
    set OPENAI_API_KEY=your_key_here
    python tests/augment_all_data.py --input all_data_fixed.csv --per-row 5
    python tests/augment_all_data.py --start 50 --limit 20
"""

import argparse
import csv
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM augmentation for NL-to-Prolog CSV datasets"
    )
    base = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--input", default=os.path.join(base, "all_data_fixed.csv"))
    p.add_argument("--output", default=os.path.join(base, "all_data_augmented.csv"))
    p.add_argument("--literals", default=os.path.join(base, "all_literals.csv"),
                   help="Path to all_literals.csv for validation context")
    p.add_argument("--model", default="gemini-3.5-flash", help="LLM model to use")
    p.add_argument("--per-row", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--no-original", action="store_true")
    p.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    return p.parse_args()


# ── OpenAI client ─────────────────────────────────────────────────────────────
def build_client(base_url: str) -> OpenAI:
    # INSERISCI LA TUA CHIAVE GEMINI QUI SOTTO TRA LE VIRGOLETTE 
    LA_MIA_CHIAVE_GEMINI = "[ENCRYPTION_KEY]"

    if LA_MIA_CHIAVE_GEMINI and not LA_MIA_CHIAVE_GEMINI.startswith("INCOLLA"):
        api_key = LA_MIA_CHIAVE_GEMINI
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")

    if base_url.strip():
        return OpenAI(base_url=base_url.strip(), api_key=api_key)
    return OpenAI()


# ── Literals loader ───────────────────────────────────────────────────────────
def load_literals(path: str) -> Dict[Tuple[str, str], List[str]]:
    """
    Load all_literals.csv and index by (domain, functor).
    Returns { ("booking", "check_in"): ["check_in(gigi, 12)", ...], ... }
    """
    idx: Dict[Tuple[str, str], List[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            domain = row["domain"].strip()
            functor = row["functor"].strip()
            literal = row["literal"].strip()
            # Remove CSV quoting
            if literal.startswith('"') and literal.endswith('"'):
                literal = literal[1:-1].replace('""', '"')
            idx.setdefault((domain, functor), []).append(literal)
    return idx


def get_domain_examples(
    literals: Dict[Tuple[str, str], List[str]],
    domain: str,
    embedding: str,
    max_examples: int = 6,
) -> str:
    """Return a few literal examples for context in the prompt."""
    functor = embedding.split("/")[0]
    key = (domain, functor)
    examples = literals.get(key, [])
    # Pick a mix: first the schema line (has uppercase args), then ground examples
    selected = examples[:max_examples]
    if not selected:
        return "(no examples available)"
    return "\n".join(f"  {ex}" for ex in selected)


# ── Prolog argument parser ───────────────────────────────────────────────────
def split_prolog_args(s: str) -> List[str]:
    """Split top-level Prolog arguments, respecting nested parens and quotes."""
    args, depth, in_q, cur = [], 0, False, ""
    for c in s:
        if c == '"' and not in_q:
            in_q = True; cur += c
        elif c == '"' and in_q:
            in_q = False; cur += c
        elif c == "(" and not in_q:
            depth += 1; cur += c
        elif c == ")" and not in_q:
            depth -= 1; cur += c
        elif c == "," and depth == 0 and not in_q:
            args.append(cur.strip()); cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    return args


def classify_arg(v: str) -> str:
    """Classify a Prolog argument value into a type tag."""
    v = v.strip()
    if v == "_" or (v and v[0].isupper()):
        return "var"
    if re.match(r"^-?\d+(\.\d+)?$", v):
        return "number"
    if v.startswith('"'):
        return "quoted"
    m = re.match(r"^([a-z_]\w*)\(", v)
    if m:
        return f"compound:{m.group(1)}"
    return f"atom"


# ── Validation ────────────────────────────────────────────────────────────────
def parse_embedding(embedding: str) -> Optional[Tuple[str, int]]:
    m = re.fullmatch(r"([a-zA-Z_]\w*)/(\d+)", embedding.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def build_type_profile(
    literals: Dict[Tuple[str, str], List[str]], domain: str, embedding: str
) -> Optional[List[Set[str]]]:
    """
    Build per-position type sets from literals.
    Returns [{"number", "var"}, {"quoted", "var"}, ...] or None.
    """
    parsed = parse_embedding(embedding)
    if not parsed:
        return None
    functor, arity = parsed
    key = (domain, functor)
    examples = literals.get(key, [])
    if not examples:
        return None

    per_pos: List[Set[str]] = [set() for _ in range(arity)]
    for ex in examples:
        ex_clean = ex.strip().strip('"').replace('""', '"')
        m = re.match(r"^[a-zA-Z0-9_]+\((.+)\)$", ex_clean, re.DOTALL)
        if not m:
            continue
        args = split_prolog_args(m.group(1))
        for i, a in enumerate(args):
            if i < arity:
                per_pos[i].add(classify_arg(a))
    return per_pos


def validate_solution(
    solution: str,
    embedding: str,
    type_profile: Optional[List[Set[str]]],
) -> Tuple[bool, str]:
    """
    Validate a generated solution against embedding + type profile.
    Returns (is_valid, reason).
    """
    parsed = parse_embedding(embedding)
    if not parsed:
        return True, "ok"
    expected_functor, expected_arity = parsed

    # Clean CSV quoting
    sol = solution.strip().strip('"').replace('""', '"')

    m = re.fullmatch(r"([a-zA-Z_]\w*)\((.*)\)\s*", sol, re.DOTALL)
    if not m:
        if expected_arity == 0 and sol == expected_functor:
            return True, "ok"
        return False, f"cannot parse solution: {sol}"

    actual_functor = m.group(1)
    args_str = m.group(2).strip()

    if actual_functor != expected_functor:
        return False, f"functor mismatch: {actual_functor} != {expected_functor}"

    args = split_prolog_args(args_str) if args_str else []
    if len(args) != expected_arity:
        return False, f"arity {len(args)} != {expected_arity}"

    # Type validation per position
    if type_profile:
        for i, arg in enumerate(args):
            if i >= len(type_profile):
                break
            arg_type = classify_arg(arg)
            if arg_type == "var":
                continue  # variables always OK
            expected_types = type_profile[i] - {"var"}
            if not expected_types:
                continue  # only vars seen → anything goes

            # Check compatibility
            compatible = False
            if arg_type in expected_types:
                compatible = True
            elif arg_type == "number" and "number" in expected_types:
                compatible = True
            elif arg_type == "quoted" and "quoted" in expected_types:
                compatible = True
            elif arg_type == "atom" and any(t == "atom" for t in expected_types):
                compatible = True
            elif arg_type.startswith("compound:"):
                # Accept any compound if examples have compounds
                if any(t.startswith("compound:") for t in expected_types):
                    compatible = True

            if not compatible:
                return False, (
                    f"arg[{i}] type '{arg_type}' incompatible with "
                    f"expected {sorted(expected_types)}"
                )

    return True, "ok"


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(
    row: Dict[str, str],
    num_variants: int,
    literals: Dict[Tuple[str, str], List[str]],
) -> str:
    sentence = row["sentence"]
    performative = row["performative"]
    embedding = row["embedding"]
    domain = row["domain"]

    # Clean solution from CSV quoting for display
    solution = row["solution"].strip()
    if solution.startswith('"') and solution.endswith('"'):
        solution = solution[1:-1].replace('""', '"')

    examples = get_domain_examples(literals, domain, embedding)

    return f"""You are an expert in NLP data augmentation for NL-to-Prolog datasets.

TASK: Given one training example, generate {num_variants} natural-language paraphrases
with correctly updated Prolog solutions.

OUTPUT FORMAT — return ONLY semicolon-separated lines, one per variant:
new_sentence;new_solution
No headers, numbering, markdown, code fences, or commentary.

═══════════════════════════════════════════════════════════════════
HARD CONSTRAINTS (violating ANY of these makes the output invalid):

1) FUNCTOR & ARITY: The solution MUST use functor "{embedding.split('/')[0]}" with
   exactly {embedding.split('/')[1]} arguments. Embedding: {embedding}.

2) PERFORMATIVE "{performative}":
   - tell → the sentence asserts a fact. All argument values come from the sentence.
   - askOne → the sentence asks for ONE answer. Known values from the sentence are
     ground; unknown/requested values use underscore "_".
   - askAll → the sentence asks for ALL matching answers. Known constraints from
     the sentence are ground; the rest use "_".

3) PROLOG SYNTAX — follow the original solution's format EXACTLY:
   - Quoted strings: if the original uses "value", your variant MUST quote that
     argument position too (e.g., "Milano", "10/11/2033"). 
     IMPORTANT: This applies to the Prolog solution ONLY! Do NOT put quotes around entities in the natural language sentence unless they were present in the original sentence.
   - Atoms: if the original uses an unquoted atom (e.g., ready, vip, cancelled),
     keep it unquoted and lowercase.
   - Compound terms: if the original uses euro(4316) or km(600), your variant
     MUST use the same compound functor for that position (e.g., euro(5000),
     dollars(300) — match the currency/unit from the sentence).
   - Numbers: never quoted, always bare integers or floats.
   - Variables/unknowns: always underscore "_".

4) SEMANTIC FIDELITY: the new sentence must express the same intent. Change
   wording, vocabulary, sentence structure — but NOT the meaning.

5) VALUE CHANGES ALLOWED: you MAY change specific values (names, dates, numbers,
   etc.) to create diversity, but the argument FORMAT must stay consistent with
   the pattern above.

6) ENGLISH ONLY. Grammatically correct. Do NOT repeat the original sentence.

═══════════════════════════════════════════════════════════════════
STYLE — write as a real user typing in a chatbot:
- Casual, direct, fluent. Vary structure genuinely.
- NEVER start with "FYI", "Update:", "For the record", "Reminder:", "Note:",
  "Heads up:", "Just so you know", "Please note", "Quick check:", "Attention:".
- NEVER prepend "Can you" / "Could you" / "Please" to an existing question
  without fully rewriting it.

═══════════════════════════════════════════════════════════════════
DOMAIN EXAMPLES (for reference on quoting/format — do NOT copy these values):
{examples}

ORIGINAL ROW:
sentence: {sentence}
performative: {performative}
solution: {solution}

Generate {num_variants} lines:"""


SYSTEM_MSG = (
    "You are a precise data augmentation engine for NL-to-Prolog datasets. "
    "Return ONLY semicolon-separated lines: sentence;solution. "
    "Preserve Prolog syntax exactly: quoting, compound terms, atoms, numbers. "
    "Never add commentary, headers, or markdown."
)


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_llm_lines(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ";" not in line:
            continue
        # Skip lines that look like headers or commentary
        if line.lower().startswith(("new_sentence", "sentence", "#", "//")):
            continue
        left, right = line.split(";", 1)
        sentence = left.strip()
        solution = right.strip()
        if sentence and solution:
            pairs.append((sentence, solution))
    return pairs


# ── API call with retry ──────────────────────────────────────────────────────
def request_variants(
    client: OpenAI,
    model: str,
    row: Dict[str, str],
    per_row: int,
    temperature: float,
    literals: Dict[Tuple[str, str], List[str]],
) -> List[Tuple[str, str]]:
    prompt = build_prompt(row, per_row, literals)
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return parse_llm_lines(content)
        except Exception as exc:
            last_exc = exc
            # Google Free Tier: 15 RPM. Se becchiamo il limite, aspettiamo sempre di più (es. 15s, 30s, 60s)
            wait = 15 * (2 ** attempt)
            print(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {exc} -- waiting {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed: {last_exc}") from last_exc


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.base_url.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Get one at https://platform.openai.com/api-keys\n"
            "Then: set OPENAI_API_KEY=sk-...\n"
            "Or use --base-url http://localhost:11434/v1 for Ollama (free)"
        )

    client = build_client(args.base_url)

    # Load literals for validation + prompt context
    literals = load_literals(args.literals)
    print(f"Loaded literals: {len(literals)} functor groups")

    with open(args.input, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Input CSV has no header")

    required = ["sentence", "performative", "embedding", "solution", "domain"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    start = max(0, args.start)
    end = len(rows) if args.limit <= 0 else min(len(rows), start + args.limit)
    selected = rows[start:end]

    print(f"Input rows  : {len(rows)}")
    print(f"Processing  : {len(selected)} (start={start}, end={end})")
    print(f"Model       : {args.model}")
    print(f"Per row     : {args.per_row}")
    print(f"Temperature : {args.temperature}")
    print()

    # ── Incremental output ────────────────────────────────────────────────────
    out_exists = os.path.isfile(args.output)
    resuming = args.start > 0 and out_exists

    out_f = open(args.output, "a" if resuming else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter=";")

    if not resuming:
        writer.writeheader()

    # Seed dedup set
    seen: Set[str] = set()
    if out_exists:
        with open(args.output, encoding="utf-8", newline="") as ef:
            for r in csv.DictReader(ef, delimiter=";"):
                seen.add(r["sentence"].strip().lower())

    # Write originals on fresh run
    if not args.no_original and not resuming:
        for row in rows:
            key = row["sentence"].strip().lower()
            if key not in seen:
                writer.writerow(row)
                seen.add(key)
        out_f.flush()

    # ── Augmentation loop ─────────────────────────────────────────────────────
    stats = {"aug": 0, "dup": 0, "invalid": 0, "errors": 0}

    for idx, row in enumerate(selected, start=start):
        print(f"[{idx + 1}/{len(rows)}] {row['performative']:7s} {row['embedding']:30s} {row['sentence'][:50]}")

        # Build type profile for this row
        type_profile = build_type_profile(literals, row["domain"], row["embedding"])

        try:
            pairs = request_variants(
                client=client,
                model=args.model,
                row=row,
                per_row=args.per_row,
                temperature=args.temperature,
                literals=literals,
            )
            for sentence, solution in pairs:
                # 1) Dedup
                key = sentence.strip().lower()
                if key in seen:
                    stats["dup"] += 1
                    continue

                # 2) Structural validation
                is_valid, reason = validate_solution(
                    solution, row["embedding"], type_profile
                )
                if not is_valid:
                    stats["invalid"] += 1
                    print(f"  [INVALID] {reason}: {solution[:60]}")
                    continue

                # 3) Write immediately
                writer.writerow({
                    "sentence": sentence,
                    "performative": row["performative"],
                    "embedding": row["embedding"],
                    "solution": solution,
                    "domain": row["domain"],
                })
                out_f.flush()
                seen.add(key)
                stats["aug"] += 1

        except Exception as exc:
            stats["errors"] += 1
            print(f"  [ERROR] row {idx + 1}: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)

    out_f.close()

    print("\n" + "=" * 60)
    print(f"Augmented rows added   : {stats['aug']}")
    print(f"Skipped (duplicate)    : {stats['dup']}")
    print(f"Skipped (invalid)      : {stats['invalid']}")
    print(f"Rows failed            : {stats['errors']}")
    print(f"Output                 : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
