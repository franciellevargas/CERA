import os
import pandas as pd
import json
import random
from textblob import TextBlob

from generate_triplets_train import build_query, chunk_text_with_offsets, overlaps

random.seed(42)

TXT_FOLDER = "txt_train"
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
N_NEGATIVES = 5
OUTPUT_FILE = "contriever_train_triplets_SUBJ_LOCAL.jsonl"


def chunk_subjectivity(text: str) -> float:
    """Compute the subjectivity score of a text chunk.

    Args:
        text (str): The text chunk to score.

    Returns:
        float: The TextBlob subjectivity score in [0.0, 1.0], or 0.0 if
            scoring fails.
    """
    try:
        return TextBlob(text).sentiment.subjectivity
    except:
        return 0.0


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

# Filter valid evidence
merged = merged[
    (merged["Evidence Start"] >= 0) &
    (merged["Evidence End"] > merged["Evidence Start"])
]

print(f"🎯 Total valid evidences: {len(merged)}")

doc_cache = {}
examples = []

print("\n🔧 Generating triplets with local subjective negatives...")

for idx, row in merged.iterrows():

    pmcid = f"PMC{row['PMCID']}"
    txt_path = os.path.join(TXT_FOLDER, f"{pmcid}.txt")

    if not os.path.exists(txt_path):
        continue

    # Load the document only once
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

    # Positives (unchanged)
    positives = [
        c for c in chunks
        if overlaps(ev_start, ev_end, c["char_start"], c["char_end"])
    ]

    if not positives:
        continue

    positive_chunk = positives[0]["text"].lower()

    # Negatives: top-k subjective
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

    # Build triplet
    example = {
        "PMCID": pmcid,
        "query": build_query(row).lower(),
        "positive": positive_chunk,
        "negatives": [n["text"].lower() for n in top_negatives]
    }

    examples.append(example)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Triplets generated: {len(examples)}")
print(f"📁 File saved: {OUTPUT_FILE}")
