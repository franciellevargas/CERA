import os
import pandas as pd
import json
import random

random.seed(42)

# =============================
# CONFIG
# =============================
TXT_FOLDER = "txt_train"
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
N_NEGATIVES = 5
OUTPUT_FILE = "contriever_train_triplets_LOCAL.jsonl"

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
print("\n🔧 Gerando triplets (local retrieval por documento)...")

for idx, row in merged.iterrows():

    pmcid_num = str(row["PMCID"])
    pmcid = f"PMC{pmcid_num}"
    txt_path = os.path.join(TXT_FOLDER, f"{pmcid}.txt")

    if not os.path.exists(txt_path):
        continue

    # carregar documento uma única vez
    if pmcid not in doc_cache:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()

        if len(doc_text.strip()) == 0:
            continue

        doc_cache[pmcid] = {
            "text": doc_text,
            "chunks": chunk_text_with_offsets(
                doc_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        }

    doc = doc_cache[pmcid]
    chunks = doc["chunks"]

    if not chunks:
        continue

    ev_start = int(row["Evidence Start"])
    ev_end = int(row["Evidence End"])

    # positivos e negativos
    positives = [
        c for c in chunks
        if overlaps(ev_start, ev_end, c["char_start"], c["char_end"])
    ]

    if not positives:
        continue

    negatives = [
        c for c in chunks
        if c not in positives
    ]

    if not negatives:
        continue

    # montar triplet
    example = {
        "PMCID": pmcid,
        "query": build_query(row).lower(),
        "positive": positives[0]["text"].lower(),
        "negatives": [
            c["text"].lower()
            for c in random.sample(
                negatives,
                k=min(N_NEGATIVES, len(negatives))
            )
        ]
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



