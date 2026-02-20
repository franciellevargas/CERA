import json
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

# =============================
# CONFIG
# =============================
MODEL_NAME = "final-contriever-finetuned-triplet+kl_maskpos51_epoch30"
JSON_FILE = "contriever_test_triplets_SUBJ_LOCAL_dedup.jsonl"
OUTPUT_JSON = "contriever_base_full_rankings.json"

MAX_LEN = 256
TOP_K_SAVE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_VALUES = [1, 3, 5, 10, 50]

# =============================
# LOAD MODEL (FROZEN)
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# =============================
# EMBEDDING
# =============================
def embed(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    emb = output.last_hidden_state[:, 0]  # CLS
    emb = F.normalize(emb, p=2, dim=1)
    return emb


# =============================
# LOAD DATA
# =============================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# =============================
# MAIN EVALUATION + SAVE RANKING
# =============================
def evaluate_and_save(data):

    recall_at_k = {k: 0 for k in K_VALUES}
    hit_at_1 = 0
    mrr_total = 0.0
    rank_distribution = Counter()

    all_rankings = []

    for ex in tqdm(data, desc=f"Evaluating ranking {MODEL_NAME}"):

        pmcid = ex.get("PMCID", "UNKNOWN")
        query = ex["query"]
        positive = ex["positive"]
        negatives = ex["negatives"]

        passages = [positive] + negatives
        labels = ["POS"] + ["NEG"] * len(negatives)

        q_emb = embed([query])
        p_emb = embed(passages)

        scores = torch.matmul(q_emb, p_emb.T).squeeze(0).tolist()

        ranked = sorted(
            zip(passages, labels, scores),
            key=lambda x: x[2],
            reverse=True
        )

        ranking = []
        positive_rank = None

        for i, (text, label, score) in enumerate(ranked, start=1):
            if label == "POS":
                positive_rank = i

            if i <= TOP_K_SAVE:
                ranking.append({
                    "rank": i,
                    "label": label,
                    "score": score,
                    "text": text
                })

        rank_distribution[positive_rank] += 1
        mrr_total += 1.0 / positive_rank

        if positive_rank == 1:
            hit_at_1 += 1

        for k in K_VALUES:
            if positive_rank <= k:
                recall_at_k[k] += 1

        all_rankings.append({
            "PMCID": pmcid,
            "query": query,
            "positive_rank": positive_rank,
            "top_50_ranking": ranking
        })

    n = len(data)

    recall_at_k = {k: v / n for k, v in recall_at_k.items()}
    hit_at_1 /= n
    mrr = mrr_total / n

    return recall_at_k, hit_at_1, mrr, rank_distribution, all_rankings


# =============================
# RUN
# =============================
if __name__ == "__main__":

    data = load_jsonl(JSON_FILE)
    print(f"Total of examples: {len(data)}")

    recall_at_k, hit_at_1, mrr, rank_dist, all_rankings = evaluate_and_save(data)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_rankings, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print(f"RESULTS – {MODEL_NAME}")
    print("==============================")

    print("\nRecall@K:")
    for k, v in recall_at_k.items():
        print(f"Recall@{k}: {v:.5f}")

    print(f"\nHit@1: {hit_at_1:.5f}")
    print(f"MRR:   {mrr:.5f}")

    print("\nPositives rank distribution:")
    for rank in sorted(rank_dist):
        print(f"Rank {rank}: {rank_dist[rank]}")

    print(f"\nComplete rankings (TOP-50) saved in: {OUTPUT_JSON}")
