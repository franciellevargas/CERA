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
OUTPUT_FILE = "contriever_train_triplets_ev_offset.jsonl"


def chunk_subjectivity(text: str) -> float:
    """Compute the subjectivity score of a text chunk.

    Args:
        text (str): The text chunk to score.

    Returns:
        float: The TextBlob subjectivity score in the range [0.0, 1.0],
            or 0.0 if scoring fails.
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

print(f"Total of valid evidences: {len(merged)}")

doc_cache = {}
examples = []

print("\nGenerating triplets with negatives using local subjective criteria...")

for idx, row in merged.iterrows():

    pmcid = f"PMC{row['PMCID']}"
    txt_path = os.path.join(TXT_FOLDER, f"{pmcid}.txt")

    if not os.path.exists(txt_path):
        continue

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

    # Positives
    positives = [
        c for c in chunks
        if overlaps(ev_start, ev_end, c["char_start"], c["char_end"])
    ]

    if not positives:
        continue

    pos_chunk_obj = positives[0]
    positive_text_raw = pos_chunk_obj["text"]
    positive_chunk = positive_text_raw.lower()

    # Evidence offsets relative to chunk (positive)
    pos_cs = pos_chunk_obj["char_start"]
    pos_ce = pos_chunk_obj["char_end"]

    evidence_start_in_chunk = max(0, ev_start - pos_cs)
    evidence_end_in_chunk = min(len(positive_text_raw), ev_end - pos_cs)

    # If for some reason there is no evidence left in positive
    if evidence_end_in_chunk <= evidence_start_in_chunk:
        continue

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

    # Triplet(+ offsets from evidence in chunk)
    example = {
        "PMCID": pmcid,
        "query": build_query(row).lower(),
        "positive": positive_chunk,
        "negatives": [n["text"].lower() for n in top_negatives],
        "evidence_start": int(evidence_start_in_chunk),
        "evidence_end": int(evidence_end_in_chunk)
    }

    examples.append(example)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\nNumber of generated triples: {len(examples)}")
print(f"Saved file: {OUTPUT_FILE}")
