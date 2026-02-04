import os
import pandas as pd
import json
import random
from textblob import TextBlob

random.seed(42)

# =============================
# CONFIG
# =============================
TXT_FOLDER = "txt_train"
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
N_NEGATIVES = 5
OUTPUT_FILE = "contriever_train_triplets_SUBJ_LOCAL.jsonl"

# =============================
# FUNÇÕES
# =============================
def build_query(row):
    return f"Compare the effect of {row['Intervention']} versus {row['Comparator']} on {row['Outcome']}."

def chunk_text_with_offsets(text, chunk_size=50, chunk_overlap=10):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    char_positions = []
    pos = 0
    for w in words:
        start = text.find(w, pos)
        end = start + len(w)
        char_positions.append((start, end))
        pos = end

    i = 0
    while i < len(words):
        sw = i
        ew = min(i + chunk_size, len(words))

        cs = char_positions[sw][0]
        ce = char_positions[ew - 1][1]

        chunks.append({
            "text": " ".join(words[sw:ew]),
            "char_start": cs,
            "char_end": ce
        })

        i += (chunk_size - chunk_overlap)

    return chunks

def overlaps(a_start, a_end, b_start, b_end):
    return not (b_end < a_start or b_start > a_end)

def chunk_subjectivity(text):
    try:
        return TextBlob(text).sentiment.subjectivity
    except:
        return 0.0

# =============================
# LOAD CSVs
# =============================
prompts = pd.read_csv("prompts_merged.csv")
annotations = pd.read_csv("annotations_merged.csv")

merged = pd.merge(
    prompts,
    annotations[
        ["PromptID", "PMCID", "Evidence Start", "Evidence End"]
    ],
    on=["PromptID", "PMCID"],
    how="inner"
)

# =============================
# FILTRAR EVIDÊNCIAS VÁLIDAS
# =============================
merged = merged[
    (merged["Evidence Start"] >= 0) &
    (merged["Evidence End"] > merged["Evidence Start"])
]

print(f"🎯 Evidências válidas totais: {len(merged)}")

# =============================
# CACHE DE DOCUMENTOS
# =============================
doc_cache = {}
examples = []

# =============================
# LOOP PRINCIPAL
# =============================
print("\n🔧 Gerando triplets com negativos subjetivos locais...")

for idx, row in merged.iterrows():

    pmcid = f"PMC{row['PMCID']}"
    txt_path = os.path.join(TXT_FOLDER, f"{pmcid}.txt")

    if not os.path.exists(txt_path):
        continue

    # carregar documento uma única vez
    if pmcid not in doc_cache:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()

        if not doc_text.strip():
            continue

        doc_cache[pmcid] = {
            "text": doc_text,
            "chunks": chunk_text_with_offsets(
                doc_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        }

    chunks = doc_cache[pmcid]["chunks"]
    if not chunks:
        continue

    ev_start = int(row["Evidence Start"])
    ev_end = int(row["Evidence End"])

    # -----------------------------
    # POSITIVO (inalterado)
    # -----------------------------
    positives = [
        c for c in chunks
        if overlaps(ev_start, ev_end, c["char_start"], c["char_end"])
    ]

    if not positives:
        continue

    positive_chunk = positives[0]["text"].lower()

    # -----------------------------
    # NEGATIVOS: TOP-K SUBJETIVOS
    # -----------------------------
    negative_candidates = [
        c for c in chunks
        if not overlaps(ev_start, ev_end, c["char_start"], c["char_end"])
    ]

    if not negative_candidates:
        continue

    scored_negatives = [
        {
            "text": c["text"],
            "subjectivity": chunk_subjectivity(c["text"])
        }
        for c in negative_candidates
    ]

    scored_negatives.sort(
        key=lambda x: x["subjectivity"],
        reverse=True
    )

    top_negatives = scored_negatives[:N_NEGATIVES]

    if len(top_negatives) < N_NEGATIVES:
        continue

    # -----------------------------
    # MONTAR TRIPLET
    # -----------------------------
    example = {
        "PMCID": pmcid,
        "query": build_query(row).lower(),
        "positive": positive_chunk,
        "negatives": [n["text"].lower() for n in top_negatives]
    }

    examples.append(example)

# =============================
# SALVAR
# =============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Triplets gerados: {len(examples)}")
print(f"📁 Arquivo salvo: {OUTPUT_FILE}")
