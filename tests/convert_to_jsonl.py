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
ALL_CSV     = os.path.join(BASE_DIR, "all_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset.jsonl")

DOMAINS = [
    "booking", "car_control", "cooking", "domestic_robot",
    "employer_management", "games_platform", "house_builder",
    "onbanking", "robot_assistant", "tickets",
]

# ── Lettura system prompt e prompt template ──────────────────────────────────
with open(SYSTEM_FILE, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open(PROMPT_FILE, encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

# ── Carica tutti i literals.csv indicizzati per dominio e functor ────────────
# Struttura: { "booking": { "booking": ["booking(anna,...)", ...], ... }, ... }
def carica_literals():
    literals = {}
    for domain in DOMAINS:
        path = os.path.join(INPUT_DIR, domain, "literals.csv")
        literals[domain] = {}
        with open(path, encoding="utf-8") as f:
            next(f)  # salta header "Literals"
            for line in f:
                lit = line.strip()
                if not lit:
                    continue
                # estrae il functor (parola prima della parentesi o dell'intera stringa)
                match = re.match(r"^([a-zA-Z0-9_]+)", lit)
                if match:
                    functor = match.group(1)
                    literals[domain].setdefault(functor, []).append(lit)
    return literals

ALL_LITERALS = carica_literals()

# ── Parser Prolog → JSON ────────────────────────────────────────────────────
def split_args(s):
    """
    Divide una stringa di argomenti Prolog per virgola,
    rispettando parentesi annidate e stringhe quotate.
    Es: 'carlo, single, "10/10/2025", "15/10/2025"' → ['carlo', 'single', '"10/10/2025"', '"15/10/2025"']
    Es: 'km(600), hour(7)' → ['km(600)', 'hour(7)']
    """
    args = []
    depth = 0       # profondità parentesi
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
    """
    Converte un singolo argomento Prolog nel valore corretto per il JSON.
    - Numeri interi  → int
    - Numeri float   → float
    - _  o variabili uppercase → "_"
    - Stringhe quoted → stringa con virgolette interne conservate
    - Atomi/termini annidati → stringa
    """
    arg = arg.strip()
    # variabile anonima o named variable (uppercase)
    if arg == "_" or (arg and arg[0].isupper()):
        return "_"
    # numero intero
    if re.match(r"^-?\d+$", arg):
        return int(arg)
    # numero float
    if re.match(r"^-?\d+\.\d+$", arg):
        return float(arg)
    # tutto il resto (atomi, stringhe quoted, termini annidati) → stringa
    return arg


def prolog_to_json(solution):
    """
    Converte un letterale Prolog nella struttura JSON attesa da jsonToTerm() in Tools.java.
    Es: booking(carlo, single, "10/10/2025", "15/10/2025")
     →  {"functor": "booking", "arg0": "carlo", "arg1": "single",
          "arg2": "\"10/10/2025\"", "arg3": "\"15/10/2025\""}
    """
    solution = str(solution).strip()

    # rimuove eventuali virgolette doppie CSV attorno all'intera soluzione
    if solution.startswith('"') and solution.endswith('"'):
        solution = solution[1:-1].replace('""', '"')

    # letterale senza argomenti: es. check_out(jonhatan) ← ha argomenti
    # letterale senza parentesi: es. "tell" (non dovrebbe capitare in solution)
    match = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", solution, re.DOTALL)
    if not match:
        # nessuna parentesi → atomo puro
        return json.dumps({"functor": solution}, ensure_ascii=False)

    functor = match.group(1)
    args_str = match.group(2)

    result = {"functor": functor}
    args = split_args(args_str)
    for i, arg in enumerate(args):
        result[f"arg{i}"] = arg_to_json_value(arg)

    return json.dumps(result, ensure_ascii=False)


# ── Costruisce NEAREST_JSON dal campo embedding (es. "booking/4") ────────────
def embedding_to_nearest_json(embedding_str, domain):
    """
    Converte "booking/4" in un literal casuale preso da literals.csv (seed fisso),
    nel formato Java HashMap.toString(): {functor=booking, arg0=X, arg1=Y, ...}
    Questo è il formato esatto prodotto da termToMap().toString() in Tools.java a runtime.
    Fallback a segnaposto numerici se non trovato alcun literal per quel functor/arità.
    """
    match = re.match(r"^([a-zA-Z0-9_]+)/(\d+)$", embedding_str.strip())
    if not match:
        return "{}"
    functor = match.group(1)
    arity   = int(match.group(2))

    # Filtra solo i literals con arità corretta, senza filtrare variabili uppercase
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
        # Formato identico a Java HashMap.toString(): {functor=X, arg0=Y, ...}
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


# ── Recupera gli esempi con lo stesso functor da literals.csv ────────────────
def get_examples(domain, embedding_str):
    """
    Restituisce la lista di esempi con lo stesso functor dal literals.csv del dominio.
    Formato: lista di stringhe come produce jsonExamples.toString() in Java.
    """
    match = re.match(r"^([a-zA-Z0-9_]+)/", embedding_str.strip())
    if not match:
        return "[]"
    functor = match.group(1)
    raw_examples = ALL_LITERALS.get(domain, {}).get(functor, [])
    # Converti ogni esempio in JSON come farebbe mapToJson() in Java
    json_examples = []
    for ex in raw_examples:
        try:
            json_examples.append(prolog_to_json(ex))
        except Exception:
            pass
    return "[" + ", ".join(json_examples) + "]"


# ── Costruisce il prompt utente compilando nl2logPrompt.txt ──────────────────
def build_user_prompt(sentence, embedding, performative, domain):
    nearest_json = embedding_to_nearest_json(embedding, domain)
    examples     = get_examples(domain, embedding)
    return (PROMPT_TEMPLATE
        .replace("SENTENCE",     sentence)
        .replace("NEAREST_JSON", nearest_json)
        .replace("ILF",          performative)
        .replace("EXAMPLES",     examples))


# ── Main: legge all_data.csv e genera dataset.jsonl ─────────────────────────
errors = 0
total  = 0

with open(ALL_CSV, encoding="utf-8") as csv_f, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:

    reader = csv.DictReader(csv_f, delimiter=";")
    for row in reader:
        try:
            user_prompt     = build_user_prompt(
                row["sentence"], row["embedding"], row["performative"], row["domain"]
            )
            assistant_reply = prolog_to_json(row["solution"])

            record = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": assistant_reply},
                ]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
        except Exception as e:
            print(f"  [ERRORE] riga {total+1} ({row.get('domain','?')}): {e} | solution={row.get('solution','?')}")
            errors += 1

print(f"\nRecords scritti : {total}")
print(f"Errori          : {errors}")
print(f"Output          : {OUTPUT_FILE}")
