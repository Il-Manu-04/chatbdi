"""
generate_test_data2.py
Genera tests/test_data2.jsonl con 10 NUOVI esempi di test nello stesso
formato di dataset.jsonl  →  {"messages": [{system}, {user}, {assistant}]}

Test cases diversi da test_data.jsonl: functor e performative differenti.

Usage (in locale):
    cd tests/
    python generate_test_data2.py
"""

import csv
import json
import re
import os
import random

random.seed(99)

# ── Percorsi ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(BASE_DIR, "input")
SYSTEM_FILE = os.path.join(BASE_DIR, "..", "interpreter", "src", "agt", "chatbdi", "modelfiles", "nl2log.txt")
PROMPT_FILE = os.path.join(BASE_DIR, "..", "interpreter", "src", "agt", "chatbdi", "modelfiles", "nl2logPrompt.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "test_data2.jsonl")

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


# ── 10 nuovi casi di test (functor diversi dal primo set, frasi originali) ───
TEST_CASES = [
    # 1. booking – fully_booked/1 – tell
    {
        "domain": "booking",
        "sentence": "The hotel has no free rooms on 24/12/2025",
        "performative": "tell",
        "embedding": "fully_booked/1",
        "solution": 'fully_booked("24/12/2025")',
    },
    # 2. car_control – fuel_level/1 – tell
    {
        "domain": "car_control",
        "sentence": "The tank is almost empty, fuel is critically low",
        "performative": "tell",
        "embedding": "fuel_level/1",
        "solution": "fuel_level(low)",
    },
    # 3. cooking – price/2 – askOne
    {
        "domain": "cooking",
        "sentence": "How expensive is meat?",
        "performative": "askOne",
        "embedding": "price/2",
        "solution": "price(meat, _)",
    },
    # 4. domestic_robot – limit/2 – tell
    {
        "domain": "domestic_robot",
        "sentence": "The maximum allowed amount of juice is 3",
        "performative": "tell",
        "embedding": "limit/2",
        "solution": "limit(juice, 3)",
    },
    # 5. employer_management – salary_details/4 – tell
    {
        "domain": "employer_management",
        "sentence": "Anna Rossi earns 48000 euros and gets paid every week",
        "performative": "tell",
        "embedding": "salary_details/4",
        "solution": 'salary_details("Anna Rossi", 48000, eur, weekly)',
    },
    # 6. games_platform – wallet_balance/2 – tell
    {
        "domain": "games_platform",
        "sentence": "User 42 has 100 dollars in their account",
        "performative": "tell",
        "embedding": "wallet_balance/2",
        "solution": "wallet_balance(42, 100)",
    },
    # 7. house_builder – resource_inventory/2 – tell
    {
        "domain": "house_builder",
        "sentence": "Right now we have 57 units of cement bags available",
        "performative": "tell",
        "embedding": "resource_inventory/2",
        "solution": 'resource_inventory("cement bags", 57)',
    },
    # 8. onbanking – failed_login/1 – tell
    {
        "domain": "onbanking",
        "sentence": "Account Marco99 could not log in",
        "performative": "tell",
        "embedding": "failed_login/1",
        "solution": 'failed_login("Marco99")',
    },
    # 9. robot_assistant – product_on_display/2 – tell
    {
        "domain": "robot_assistant",
        "sentence": "There is a SmartWatch on the wearables display",
        "performative": "tell",
        "embedding": "product_on_display/2",
        "solution": 'product_on_display("SmartWatch", wearables)',
    },
    # 10. tickets – refund_request/3 – tell
    {
        "domain": "tickets",
        "sentence": "Order 55 refund was submitted on 5/5/2023 and it is still pending",
        "performative": "tell",
        "embedding": "refund_request/3",
        "solution": 'refund_request(55, "5/5/2023", pending)',
    },
]

# ── Genera test_data2.jsonl ──────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for i, tc in enumerate(TEST_CASES, 1):
        user_prompt     = build_user_prompt(
            tc["sentence"], tc["embedding"], tc["performative"], tc["domain"]
        )
        assistant_reply = prolog_to_json(tc["solution"])
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

print(f"\nGenerati {len(TEST_CASES)} record in {OUTPUT_FILE}")
