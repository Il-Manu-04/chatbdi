"""
generate_test_data.py
Genera tests/test_data.jsonl con 10 esempi di test nello STESSO
formato di dataset.jsonl  →  {"messages": [{system}, {user}, {assistant}]}

Usage (in locale):
    cd tests/
    python generate_test_data.py
Poi carica test_data.jsonl su Drive in Tirocinio_bechelor/
"""

import csv
import json
import re
import os
import random

random.seed(42)

# ── Percorsi ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(BASE_DIR, "input")
SYSTEM_FILE = os.path.join(BASE_DIR, "..", "interpreter", "src", "agt", "chatbdi", "modelfiles", "nl2log.txt")
PROMPT_FILE = os.path.join(BASE_DIR, "..", "interpreter", "src", "agt", "chatbdi", "modelfiles", "nl2logPrompt.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "test_data.jsonl")

DOMAINS = [
    "booking", "car_control", "cooking", "domestic_robot",
    "employer_management", "games_platform", "house_builder",
    "onbanking", "robot_assistant", "tickets",
]

# ── System prompt e prompt template ─────────────────────────────────────────
with open(SYSTEM_FILE, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open(PROMPT_FILE, encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

# ── Carica literals.csv per tutti i domini ───────────────────────────────────
def carica_literals():
    literals = {}
    for domain in DOMAINS:
        path = os.path.join(INPUT_DIR, domain, "literals.csv")
        literals[domain] = {}
        with open(path, encoding="utf-8") as f:
            next(f)  # salta header
            for line in f:
                lit = line.strip()
                if not lit:
                    continue
                match = re.match(r"^([a-zA-Z0-9_]+)", lit)
                if match:
                    functor = match.group(1)
                    literals[domain].setdefault(functor, []).append(lit)
    return literals

ALL_LITERALS = carica_literals()

# ── Parser Prolog → JSON (identico a convert_to_jsonl.py) ───────────────────
def split_args(s):
    args = []
    depth = 0
    in_quote = False
    current = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"' and not in_quote:
            in_quote = True
            current += c
        elif c == '"' and in_quote:
            in_quote = False
            current += c
        elif c == '(' and not in_quote:
            depth += 1
            current += c
        elif c == ')' and not in_quote:
            depth -= 1
            current += c
        elif c == ',' and depth == 0 and not in_quote:
            args.append(current.strip())
            current = ""
        else:
            current += c
        i += 1
    if current.strip():
        args.append(current.strip())
    return args


def arg_to_json_value(arg):
    arg = arg.strip()
    if arg == "_" or (arg and arg[0].isupper()):
        return "_"
    if re.match(r"^-?\d+$", arg):
        return int(arg)
    if re.match(r"^-?\d+\.\d+$", arg):
        return float(arg)
    return arg


def prolog_to_json(solution):
    solution = str(solution).strip()
    if solution.startswith('"') and solution.endswith('"'):
        solution = solution[1:-1].replace('""', '"')
    match = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", solution, re.DOTALL)
    if not match:
        return json.dumps({"functor": solution}, ensure_ascii=False)
    functor = match.group(1)
    args_str = match.group(2)
    result = {"functor": functor}
    args = split_args(args_str)
    for i, arg in enumerate(args):
        result[f"arg{i}"] = arg_to_json_value(arg)
    return json.dumps(result, ensure_ascii=False)


# ── Costruisce NEAREST_JSON dal campo embedding ──────────────────────────────
def embedding_to_nearest_json(embedding_str, domain):
    match = re.match(r"^([a-zA-Z0-9_]+)/(\d+)$", embedding_str.strip())
    if not match:
        return "{}"
    functor = match.group(1)
    arity   = int(match.group(2))

    candidates = []
    for ex in ALL_LITERALS.get(domain, {}).get(functor, []):
        m = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", ex.strip(), re.DOTALL)
        if not m:
            continue
        args = split_args(m.group(2))
        if len(args) == arity:
            candidates.append((ex, args))

    if candidates:
        ex, args = random.choice(candidates)
        parts = [f"functor={functor}"]
        for i, arg in enumerate(args):
            parts.append(f"arg{i}={arg_to_json_value(arg)}")
        return "{" + ", ".join(parts) + "}"

    # fallback: segnaposto numerici
    d = {"functor": functor}
    for i in range(arity):
        d[f"arg{i}"] = i + 1
    parts = ", ".join([f"{k}={v}" for k, v in d.items()])
    return "{" + parts + "}"


# ── Recupera esempi dal literals.csv ────────────────────────────────────────
def get_examples(domain, embedding_str):
    match = re.match(r"^([a-zA-Z0-9_]+)/", embedding_str.strip())
    if not match:
        return "[]"
    functor = match.group(1)
    raw_examples = ALL_LITERALS.get(domain, {}).get(functor, [])
    json_examples = []
    for ex in raw_examples:
        try:
            json_examples.append(prolog_to_json(ex))
        except Exception:
            pass
    return "[" + ", ".join(json_examples) + "]"


# ── Costruisce il prompt utente ──────────────────────────────────────────────
def build_user_prompt(sentence, embedding, performative, domain):
    nearest_json = embedding_to_nearest_json(embedding, domain)
    examples     = get_examples(domain, embedding)
    return (PROMPT_TEMPLATE
        .replace("SENTENCE",     sentence)
        .replace("NEAREST_JSON", nearest_json)
        .replace("ILF",          performative)
        .replace("EXAMPLES",     examples))


# ── 10 casi di test (uno per dominio, NON presenti nel training set) ─────────
# Formato: sentence, performative, embedding, solution (Prolog literal)
TEST_CASES = [
    {
        "domain": "booking",
        "sentence": "Marco booked a suite from 20/07/2025 to 25/07/2025",
        "performative": "tell",
        "embedding": "booking/4",
        "solution": 'booking(marco, suite, "20/07/2025", "25/07/2025")',
    },
    {
        "domain": "car_control",
        "sentence": "The passenger door is still open",
        "performative": "tell",
        "embedding": "door_status/2",
        "solution": "door_status(passenger, open)",
    },
    {
        "domain": "cooking",
        "sentence": "What are all the ingredients of pasta carbonara?",
        "performative": "askAll",
        "embedding": "ingredients/2",
        "solution": "ingredients(pasta(carbonara), _)",
    },
    {
        "domain": "domestic_robot",
        "sentence": "There is some milk in the fridge",
        "performative": "tell",
        "embedding": "available/2",
        "solution": "available(milk, fridge)",
    },
    {
        "domain": "employer_management",
        "sentence": "What certifications does Michael Brown hold?",
        "performative": "askAll",
        "embedding": "certification_details/4",
        "solution": 'certification_details("Michael Brown", _, _, _)',
    },
    {
        "domain": "games_platform",
        "sentence": "User 77 and user 88 are friends",
        "performative": "tell",
        "embedding": "user_friend/2",
        "solution": "user_friend(77, 88)",
    },
    {
        "domain": "house_builder",
        "sentence": "The cost of Electrical work for project 101 is 2500 euros",
        "performative": "tell",
        "embedding": "budget_item/3",
        "solution": 'budget_item(101, "Electrical", 2500)',
    },
    {
        "domain": "onbanking",
        "sentence": "User Pippo02 has logged in successfully",
        "performative": "tell",
        "embedding": "successful_login/1",
        "solution": 'successful_login("Pippo02")',
    },
    {
        "domain": "robot_assistant",
        "sentence": "There are no more Tablets available",
        "performative": "tell",
        "embedding": "out_of_stock/1",
        "solution": 'out_of_stock("Tablet")',
    },
    {
        "domain": "tickets",
        "sentence": "Which events currently have ready status?",
        "performative": "askAll",
        "embedding": "event_status/2",
        "solution": "event_status(_, ready)",
    },
]

# ── Genera test_data.jsonl (stesso formato di dataset.jsonl) ────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for i, tc in enumerate(TEST_CASES, 1):
        user_prompt      = build_user_prompt(
            tc["sentence"], tc["embedding"], tc["performative"], tc["domain"]
        )
        assistant_reply  = prolog_to_json(tc["solution"])
        record = {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": assistant_reply},
            ]
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [{i:2d}] {tc['domain']:25s} | {tc['sentence'][:55]}")
        print(f"         expected → {assistant_reply}")

print(f"\nGenerati {len(TEST_CASES)} record → {OUTPUT_FILE}")
