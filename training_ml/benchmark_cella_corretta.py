import json
import torch
import re
import time
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
from unsloth import FastLanguageModel
from google.colab import drive

# ==========================================
# 1. MONTAGGIO DRIVE E PERCORSI
# ==========================================
drive.mount('/content/drive')

# Cartella dove caricare gli 8 file .jsonl di test e dove verranno salvati i risultati
DRIVE_FOLDER = "/content/drive/MyDrive/Tesi_BDI/"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

# Percorso degli adattatori LoRA sul Drive (Unsloth legge adapter_config.json
# e scarica automaticamente il modello base Qwen3.5-4B da internet)
PATH_MODELLO_FINETUNED = "/content/drive/MyDrive/Tirocinio_bechelor/lora_chatbdi_qwen3.5-4B"

# ==========================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================
def normalize(obj):
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, str):
            v = v.strip()
            if v == "_" or (v and v[0].isupper() and v.replace("_", "").isalpha()):
                v = "__VAR__"
        out[k] = v
    return out

def fix_hallucinated_json(text):
    """
    Prende output sporco (da Base o FT) e restituisce una stringa JSON perfetta,
    aggiungendo virgolette mancanti a chiavi e argomenti.
    """
    # 1. Cambia = in : per il Fine-Tuned
    text = re.sub(r'([a-zA-Z0-9_]+)\s*=', r'\1: ', text)
    
    # 2. Usa YAML per estrarre i dati ignorando virgolette mancanti (es. functor: user_account)
    try:
        dizionario = yaml.safe_load(text)
    except Exception:
        return text
        
    if not isinstance(dizionario, dict):
        return text
        
    # 3. Aggiunge virgolette interne agli 'arg' se il modello le ha scordate
    for k, v in dizionario.items():
        if isinstance(v, str) and k.startswith("arg"):
            # Ignora la variabile anonima
            if v == "_":
                continue
            # Se la parola NON ha già le virgolette interne, le aggiunge
            if not (v.startswith('"') and v.endswith('"')):
                dizionario[k] = f'"{v}"'
                
    # 4. Restituisce un JSON impeccabile
    return json.dumps(dizionario)

def calcola_accuratezza_parziale(expected_dict, predicted_dict):
    if not isinstance(expected_dict, dict) or not isinstance(predicted_dict, dict):
        return 0.0
    totale_chiavi = len(expected_dict)
    if totale_chiavi == 0:
        return 1.0 if len(predicted_dict) == 0 else 0.0
    chiavi_corrette = sum(1 for k, v in expected_dict.items() if k in predicted_dict and predicted_dict[k] == v)
    return chiavi_corrette / totale_chiavi




class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = set([s for s in stop_ids if s is not None])
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1].item() in self.stop_ids

# ==========================================
# 3. CONFIGURAZIONE ESPERIMENTI
# ==========================================
# [FIX 1] Modello base corretto: Qwen2.5-Coder 7B (quello usato in produzione via Ollama)
# [FIX 2] Domini corretti:
#   - ticket  = Out-of-Domain (NON presente nei dati di training)
#   - ignoto  = In-Domain     (usa frasi nuove dai 9 domini di training noti)

esperimenti = [
    {
        "nome_modello": "Qwen2.5-Coder-7B (Base)",
        "path": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",  # Scaricato da internet
        "trained": "No",
        "datasets": [
            {"id": "base_ticket_zero.jsonl", "domain": "Out-of-Domain", "esempi": "Zero-Shot"},
            {"id": "base_ticket_few.jsonl",  "domain": "Out-of-Domain", "esempi": "Few-Shot"},
            {"id": "base_ignoto_zero.jsonl", "domain": "In-Domain",     "esempi": "Zero-Shot"},
            {"id": "base_ignoto_few.jsonl",  "domain": "In-Domain",     "esempi": "Few-Shot"}
        ]
    },
    {
        "nome_modello": "Qwen3.5-4B (Fine-Tuned)",
        "path": PATH_MODELLO_FINETUNED,  # Adattatori LoRA dal Drive
        "trained": "Yes",
        "datasets": [
            {"id": "ft_ticket_zero.jsonl", "domain": "Out-of-Domain", "esempi": "Zero-Shot"},
            {"id": "ft_ticket_few.jsonl",  "domain": "Out-of-Domain", "esempi": "Few-Shot"},
            {"id": "ft_ignoto_zero.jsonl", "domain": "In-Domain",     "esempi": "Zero-Shot"},
            {"id": "ft_ignoto_few.jsonl",  "domain": "In-Domain",     "esempi": "Few-Shot"}
        ]
    }
]

# ==========================================
# 4. INFERENZA MASSIVA E BENCHMARK
# ==========================================
risultati_globali = []

for config_mod in esperimenti:
    print(f"\n🚀 CARICAMENTO: {config_mod['nome_modello']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config_mod['path'],
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    im_end_id = text_tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_crit = StoppingCriteriaList([StopOnTokens([text_tokenizer.eos_token_id, im_end_id])])

    for ds_config in config_mod['datasets']:
        file_path_drive = os.path.join(DRIVE_FOLDER, ds_config['id'])

        if not os.path.exists(file_path_drive):
            print(f"❌ SALTO: {ds_config['id']} non trovato su Drive.")
            continue

        with open(file_path_drive, encoding="utf-8") as f:
            test_cases = [json.loads(line) for line in f if line.strip()]

        for i, tc in enumerate(tqdm(test_cases, desc=f"Test {ds_config['id']}")):
            system = tc["messages"][0]["content"]
            user = tc["messages"][1]["content"]
            expected = tc["messages"][2]["content"]

            prompt_testuale = text_tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=False
            )
            prompt_testuale += "<|im_start|>assistant\n{"

            tokens = text_tokenizer(prompt_testuale, return_tensors="pt")
            input_ids = tokens["input_ids"].to("cuda")
            attention_mask = tokens["attention_mask"].to("cuda") if "attention_mask" in tokens else torch.ones_like(input_ids)
            prompt_len = input_ids.shape[1]

            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = 128,
                    do_sample = False,
                    pad_token_id = text_tokenizer.eos_token_id,
                    stopping_criteria = stop_crit,
                )
            end_time = time.time()

            latenza_sec = end_time - start_time
            token_generati = len(output_ids[0]) - prompt_len
            tps = token_generati / latenza_sec if latenza_sec > 0 else 0

            predicted_raw = text_tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
            predicted_raw = "{" + predicted_raw

            cleaned = predicted_raw
            # Fix per non far crashare l'interfaccia con i backtick
            backticks = "`" * 3
            if cleaned.startswith(backticks + "json"): cleaned = cleaned[7:]
            elif cleaned.startswith(backticks): cleaned = cleaned[3:]
            if cleaned.endswith(backticks): cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # [MODIFICA 1] Applichiamo il fixer a prescindere dal modello se il JSON è rotto
            try:
                json.loads(cleaned)
            except json.JSONDecodeError:
                cleaned = fix_hallucinated_json(cleaned)

            passed = False
            parziale = 0.0
            try:
                dict_atteso = normalize(json.loads(expected))
                dict_previsto = normalize(json.loads(cleaned))
                passed = (dict_atteso == dict_previsto)
                parziale = calcola_accuratezza_parziale(dict_atteso, dict_previsto)
            except json.JSONDecodeError:
                pass

            risultati_globali.append({
                "Modello": config_mod['nome_modello'],
                "Addestrato": config_mod['trained'],
                "Dominio": ds_config['domain'],
                "Strategia": ds_config['esempi'],
                "Test_ID": i + 1,
                "Prompt_Tokens": prompt_len,
                "Latenza_Sec": round(latenza_sec, 3),
                "Tokens_Per_Sec": round(tps, 2),
                "Exact_Match": 1 if passed else 0,
                "Slot_Filling_Acc": round(parziale, 3),
                "Output_Grezzo": cleaned
            })

    print(f"🧹 Pulizia VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 5. SALVATAGGIO CSV E GENERAZIONE GRAFICI
# ==========================================
csv_save_path = os.path.join(DRIVE_FOLDER, "benchmark_completo_tesi.csv")
df = pd.DataFrame(risultati_globali)
df.to_csv(csv_save_path, index=False)
print(f"\n✅ DATI SALVATI IN: {csv_save_path}")

print("📊 Generazione dei grafici in corso...")
df["Exact_Match_Pct"] = df["Exact_Match"] * 100
df["Slot_Filling_Pct"] = df["Slot_Filling_Acc"] * 100
agg_df = df.groupby(["Modello", "Dominio", "Strategia"])[["Exact_Match_Pct", "Slot_Filling_Pct", "Latenza_Sec"]].mean().reset_index()

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Grafico 1: Exact Match vs Slot-Filling per il modello Fine-Tuned
ft_data = agg_df[agg_df["Modello"] == "Qwen3.5-4B (Fine-Tuned)"].melt(
    id_vars=["Dominio", "Strategia"], value_vars=["Exact_Match_Pct", "Slot_Filling_Pct"],
    var_name="Metrica", value_name="Punteggio"
)
ft_data["Metrica"] = ft_data["Metrica"].replace({"Exact_Match_Pct": "Exact Match (100% Corretto)", "Slot_Filling_Pct": "Slot-Filling (Parziale)"})
plt.figure(figsize=(10, 6))
sns.barplot(data=ft_data, x="Strategia", y="Punteggio", hue="Metrica", palette="Set2")
plt.title("Qwen3.5 4B FT: Exact Match vs Slot-Filling Accuracy", fontweight="bold")
plt.ylabel("Accuratezza Media (%)")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_FOLDER, "1_Metriche_Confronto.pdf"), dpi=300)
plt.close()

# Grafico 2: Heatmap di Generalizzazione
plt.figure(figsize=(8, 5))
pivot_table = agg_df[agg_df["Modello"] == "Qwen3.5-4B (Fine-Tuned)"].pivot_table(values="Slot_Filling_Pct", index="Strategia", columns="Dominio")
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100)
plt.title("Generalizzazione (Slot-Filling %)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_FOLDER, "2_Generalizzazione.pdf"), dpi=300)
plt.close()

# Grafico 3: Latenza per entrambi i modelli
plt.figure(figsize=(9, 5))
sns.barplot(data=agg_df, x="Modello", y="Latenza_Sec", hue="Strategia", palette="magma")
plt.title("Tempo di Generazione per Query", fontweight="bold")
plt.ylabel("Secondi")
plt.tight_layout()
plt.savefig(os.path.join(DRIVE_FOLDER, "3_Latenza_Confronto.pdf"), dpi=300)
plt.close()

print("✅ TUTTO COMPLETATO! CSV e PDF sono salvati nel tuo Drive.")
