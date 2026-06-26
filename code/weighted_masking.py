import json
import spacy
import torch

from transformers import AutoTokenizer

MODEL_NAME = "facebook/contriever"
MAX_LENGTH = 128

JSON_FILE = "contriever_train_triplets_ev_offset_dedup.jsonl"

POS_WEIGHTS = {
    "NOUN": 1.5,
    "PROPN": 0.5,
    "VERB": 0.8,
    "ADJ": 0.9,
    "ADV": 0.5,
    "NUM": 0.6,

    "PRON": 0.1,
    "DET": 0.0,
    "ADP": 0.1,
    "AUX": 0.1,
    "CCONJ": 0.0,
    "SCONJ": 0.0,
    "PART": 0.1,

    "PUNCT": 0.0,
    "SPACE": 0.0,

    "X": 0.3
}

''' # Uncomment to use
POS_WEIGHTS = {
    "NOUN": 1.0,
    "PROPN": 1.0,
    "VERB": 1.0,
    "ADJ": 0.9,
    "ADV": 0.8,
    "NUM": 0.7,
    "PRON": 0.4,
    "DET": 0.2,
    "ADP": 0.3,
    "AUX": 0.4,
    "CCONJ": 0.2,
    "SCONJ": 0.2,
    "PART": 0.5,
    "PUNCT": 0.0,
    "SPACE": 0.0,
    "X": 0.5
}
'''

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

with open(JSON_FILE, "r", encoding="utf-8") as f:

    for example_idx, line in enumerate(f):

        if example_idx >= 3:
            break

        item = json.loads(line.strip())

        positive = item["positive"]

        ev_s = int(item["evidence_start"])
        ev_e = int(item["evidence_end"])

        print("\n" + "=" * 120)
        print(f"EXAMPLE {example_idx}")
        print("=" * 120)

        print("\nFULL POSITIVE:\n")
        print(positive)

        print("\nEVIDENCE:\n")
        print(positive[ev_s:ev_e])

        # spaCy tokenization
        doc = nlp(positive)

        spacy_tokens = []

        for token in doc:
            spacy_tokens.append({
                "text": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text),
                "pos": token.pos_
            })

        # Transformer tokenization
        enc = tokenizer(
            positive,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        rat_mask = torch.zeros_like(attn, dtype=torch.float)

        print("\nTOKEN ANALYSIS:\n")

        for i in range(len(tokens)):

            tok = tokens[i]

            ts = int(offsets[i, 0].item())
            te = int(offsets[i, 1].item())

            # Special token
            if ts == 0 and te == 0:

                print(
                    f"{i:03d} | "
                    f"{tok:<15} | "
                    f"SPECIAL TOKEN"
                )

                continue

            # Intersects evidence?
            intersects = not (te <= ev_s or ts >= ev_e)

            overlapping_pos = []

            for sp_tok in spacy_tokens:

                sp_s = sp_tok["start"]
                sp_e = sp_tok["end"]

                overlap = not (te <= sp_s or ts >= sp_e)

                if overlap:
                    overlapping_pos.append(sp_tok["pos"])

            if len(overlapping_pos) == 0:
                pos_tag = "X"
            else:
                pos_tag = max(
                    set(overlapping_pos),
                    key=overlapping_pos.count
                )

            weight = 0.0

            if intersects:
                weight = POS_WEIGHTS.get(pos_tag, 0.3)
                rat_mask[i] = weight

            text_span = positive[ts:te]

            print(
                f"{i:03d} | "
                f"{tok:<15} | "
                f"span=({ts:3d},{te:3d}) | "
                f"text='{text_span:<20}' | "
                f"POS={pos_tag:<8} | "
                f"in_ev={str(intersects):<5} | "
                f"weight={weight:.2f}"
            )

        print("\nRATIONALE MASK:\n")
        print(rat_mask.tolist())

        print("\nNON-ZERO WEIGHTS:\n")

        for i in range(len(rat_mask)):

            if rat_mask[i].item() > 0:

                print(
                    f"{tokens[i]:<15} -> {rat_mask[i].item():.2f}"
                )

        print("\n" + "=" * 120)
