import os
import pandas as pd
import json
import random

random.seed(42)

TXT_FOLDER = "txt_train"
CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
N_NEGATIVES = 5
OUTPUT_FILE = "contriever_train_triplets_LOCAL.jsonl"


def build_query(row: pd.Series) -> str:
    """Build a natural-language comparison query from an annotation row.

    Args:
        row (pd.Series): A row containing the Intervention, Comparator
            and Outcome fields.

    Returns:
        str: A query of the form "Compare the effect of <Intervention> versus
            <Comparator> on <Outcome>.".
    """
    return f"Compare the effect of {row['Intervention']} versus {row['Comparator']} on {row['Outcome']}."


def chunk_text_with_offsets(text: str, chunk_size: int = 50, chunk_overlap: int = 10) -> list[dict]:
    """Split text into overlapping word chunks while tracking character offsets.

    Args:
        text (str): The document text to chunk.
        chunk_size (int): Number of words per chunk. Defaults to 50.
        chunk_overlap (int): Number of words shared between consecutive chunks.
            Defaults to 10.

    Returns:
        list[dict]: A list of chunks, each a dict with keys text (str),
            char_start (int) and char_end (int) marking the chunk's
            character span in the original text. Empty if text has no words.
    """
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


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check whether two character spans overlap.

    Args:
        a_start (int): Start offset of the first span.
        a_end (int): End offset of the first span.
        b_start (int): Start offset of the second span.
        b_end (int): End offset of the second span.

    Returns:
        bool: True if the spans [a_start, a_end] and [b_start, b_end] overlap,
            False otherwise.
    """
    return not (b_end < a_start or b_start > a_end)


if __name__ == "__main__":
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

    print("\n🔧 Generating triplets (local retrieval per document)...")

    for idx, row in merged.iterrows():

        pmcid_num = str(row["PMCID"])
        pmcid = f"PMC{pmcid_num}"
        txt_path = os.path.join(TXT_FOLDER, f"{pmcid}.txt")

        if not os.path.exists(txt_path):
            continue

        # Load the document only once
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

        # Triplet
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Triplets generated: {len(examples)}")
    print(f"📁 File saved: {OUTPUT_FILE}")
