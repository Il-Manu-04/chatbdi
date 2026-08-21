# =============================================================================
# CONTEGGIO TOKEN DEI DATASET DI ADDESTRAMENTO
# =============================================================================
# COME USARLO:
#   1. Aprire https://colab.new  (crea un notebook nuovo e vuoto)
#   2. Incollare TUTTO questo codice in un'unica cella
#   3. Eseguire la cella
#   4. Autorizzare il montaggio di Google Drive quando richiesto
#   5. Se compare la richiesta di caricamento, selezionare dal PC i file
#      mancanti (si trovano in locale nella cartella chatbdi/tests/)
#
# NON eseguirlo dentro 01_Training_LoRA.ipynb: si rischia di far ripartire
# l'addestramento eseguendo per sbaglio le celle precedenti.
# =============================================================================

import json
import os
import statistics as st

from google.colab import drive, files
from transformers import AutoTokenizer

# ── Configurazione ───────────────────────────────────────────────────────────
ROOT = "/content/drive/MyDrive/Tirocinio_bechelor"

# (nome file, descrizione, max_seq_length usato all'epoca)
DATASET = [
    ("dataset.jsonl", "prompt lungo 7615 car. - prove preliminari", 3072),
    ("dataset_fixed.jsonl", "prompt lungo 9185 car.", 3072),
    ("dataset_short_R2.jsonl", "prompt breve 901 car. - DEFINITIVO", 512),
]

# I due tokenizer hanno vocabolari diversi: le lunghezze non sono confrontabili
# fra tokenizer, ma lo sono fra dataset a parita' di tokenizer.
TOKENIZER = [
    ("Qwen3.5-4B (modello finale)", "unsloth/Qwen3.5-4B"),
    ("Qwen2.5-Coder-3B (prove preliminari)", "unsloth/Qwen2.5-Coder-3B-Instruct"),
]

# ── 1. Montaggio Drive ───────────────────────────────────────────────────────
drive.mount("/content/drive")

# ── 2. Ricerca ricorsiva dei file dentro la cartella del tirocinio ───────────
attesi = [nome for nome, _, _ in DATASET]
percorso = {}

if os.path.isdir(ROOT):
    for cartella, _, contenuto in os.walk(ROOT):
        for nome in contenuto:
            if nome in attesi and nome not in percorso:
                percorso[nome] = os.path.join(cartella, nome)
else:
    print(f"ATTENZIONE: la cartella {ROOT} non esiste. Controllare il nome.")

print("\nRicerca su Drive:")
for nome in attesi:
    print(f"  {nome:24} -> {percorso.get(nome, 'NON TROVATO')}")

# ── 3. Caricamento manuale dei file mancanti ─────────────────────────────────
mancanti = [n for n in attesi if n not in percorso]
if mancanti:
    print(f"\nMancano {len(mancanti)} file: {', '.join(mancanti)}")
    print("Si trovano in locale in chatbdi/tests/ — selezionarli qui sotto.")
    print("(Per saltare, premere Annulla: verranno semplicemente ignorati.)\n")
    try:
        caricati = files.upload()
        for nome in caricati:
            percorso[nome] = f"/content/{nome}"
    except Exception as e:
        print(f"Caricamento saltato ({e})")


# ── 4. Funzioni di calcolo ───────────────────────────────────────────────────
def carica(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(r) for r in f if r.strip()]


def conta_token(esempi, tok):
    """Applica il template conversazionale e misura la lunghezza in token."""
    lunghezze = []
    for e in esempi:
        testo = tok.apply_chat_template(
            e["messages"], tokenize=False, add_generation_prompt=False
        )
        lunghezze.append(len(tok(testo)["input_ids"]))
    return lunghezze


# ── 5. Esecuzione ────────────────────────────────────────────────────────────
for etichetta_tok, nome_tok in TOKENIZER:
    print("\n" + "=" * 72)
    print(f"TOKENIZER: {etichetta_tok}")
    print("=" * 72)

    try:
        tok = AutoTokenizer.from_pretrained(nome_tok)
    except Exception as e:
        print(f"  Non caricabile: {e}")
        continue

    for nome, descrizione, limite in DATASET:
        if nome not in percorso:
            print(f"\n  {nome}: non disponibile, saltato")
            continue

        esempi = carica(percorso[nome])
        caratteri = [sum(len(m["content"]) for m in e["messages"]) for e in esempi]
        lung = sorted(conta_token(esempi, tok))
        n = len(lung)
        oltre = sum(1 for x in lung if x > limite)

        print(f"\n  {nome}  ({descrizione})")
        print(f"    esempi ..................... {n}")
        print(f"    token mediana .............. {st.median(lung):.0f}")
        print(f"    token media ................ {st.mean(lung):.1f}")
        print(f"    token min / max ............ {lung[0]} / {lung[-1]}")
        print(f"    token 95o percentile ....... {lung[int(0.95 * (n - 1))]}")
        print(f"    max_seq_length dell'epoca .. {limite}")
        print(f"    esempi oltre il limite ..... {oltre} ({oltre / n * 100:.1f}%)")
        print(f"    caratteri per token ........ {sum(caratteri) / sum(lung):.2f}")

print("\n\nFATTO. Copiare l'output e incollarlo nella chat.")
