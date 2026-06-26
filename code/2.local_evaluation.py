import json
import math
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter, defaultdict

K_VALUES = [1, 3, 5, 10, 20, 50]
TOP_K_SAVE = 50
MAX_LEN = 128
BATCH_SIZE_PASSAGES = 64
BATCH_SIZE_QUERIES = 1

MODEL_NAME = "facebook/contriever"
JSON_FILE = "test_local_grouped.jsonl"
#OUTPUT_JSON = f"{MODEL_NAME.strip().split('/')[-1]}_pmcid_local_top_{max(K_VALUES)}_rankings.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def load_jsonl(path: str) -> list:
    """Load a JSON Lines file into a list of parsed records.

    Args:
        path (str): Path to the .jsonl file to read. Blank lines are ignored.

    Returns:
        list: One parsed object (typically a dict) per non-empty line.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedup_preserve_order(items: list) -> list:
    """Remove duplicate items while keeping their first-seen order.

    Args:
        items (list): Iterable of hashable items that may contain duplicates.

    Returns:
        list: Items with duplicates removed, preserving the original order.
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@torch.no_grad()
def embed_texts(texts: list, batch_size: int = 64) -> torch.Tensor:
    """Encode texts into L2-normalized embeddings using the global model.

    Texts are tokenized and embedded in batches; the [CLS] token of the last
    hidden state is used as the representation and then L2-normalized.

    Args:
        texts (list): Strings to embed.
        batch_size (int): Number of texts encoded per forward pass.

    Returns:
        torch.Tensor: Tensor of shape (len(texts), hidden_size) on CPU
            containing the normalized embeddings.
    """
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        output = model(**encoded)
        emb = output.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=1)

        embs.append(emb.detach().cpu())

    return torch.cat(embs, dim=0)


def topk_from_scores(scores_1d: torch.Tensor, k: int) -> tuple:
    """Return the indices and values of the top-k highest scores.

    Args:
        scores_1d (torch.Tensor): 1-D tensor of scores.
        k (int): Number of top entries to return. Capped at the number of
            available scores.

    Returns:
        tuple: A (indices, values) pair, where indices is a list of
            int positions and values is a list of float scores, both
            ordered from highest to lowest score.
    """
    vals, idxs = torch.topk(
        scores_1d,
        k=min(k, scores_1d.shape[0]),
        largest=True
    )
    return idxs.tolist(), vals.tolist()


def dcg_at_k(binary_relevance: list) -> float:
    """Compute Discounted Cumulative Gain over a relevance sequence.

    Args:
        binary_relevance (list): Relevance grades ordered by rank (typically
            0/1 values), where index 0 is the top-ranked item.

    Returns:
        float: The DCG value, where each positive relevance contributes
            rel / log2(rank + 1).
    """
    dcg = 0.0
    for i, rel in enumerate(binary_relevance, start=1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k_from_ranking(ranked_indices: list, positive_index_set: set, k: int) -> float:
    """Compute normalized DCG at k for a single ranking.

    Args:
        ranked_indices (list): Passage indices ordered from most to least
            relevant according to the model.
        positive_index_set (set): Indices that count as relevant (positives).
        k (int): Cutoff rank for the evaluation.

    Returns:
        float: NDCG@k in the range [0.0, 1.0]; 0.0 when the ideal DCG
            is zero (no positives within reach).
    """
    top_k = ranked_indices[:k]
    binary_rel = [1 if idx in positive_index_set else 0 for idx in top_k]
    dcg = dcg_at_k(binary_rel)

    ideal_hits = min(len(positive_index_set), k)
    ideal_rel = [1] * ideal_hits
    idcg = dcg_at_k(ideal_rel)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(ranked_indices: list, positive_index_set: set, k: int) -> float:
    """Compute Average Precision at k for a single ranking.

    Args:
        ranked_indices (list): Passage indices ordered from most to least
            relevant according to the model.
        positive_index_set (set): Indices that count as relevant (positives).
        k (int): Cutoff rank for the evaluation.

    Returns:
        float: AP@k in the range [0.0, 1.0], normalized by
            min(num_positives, k); 0.0 when there are no positives.
    """
    top_k = ranked_indices[:k]
    hits = 0
    precision_sum = 0.0

    for rank, idx in enumerate(top_k, start=1):
        if idx in positive_index_set:
            hits += 1
            precision_sum += hits / rank

    denom = min(len(positive_index_set), k)
    if denom == 0:
        return 0.0
    return precision_sum / denom


def reciprocal_rank(ranked_indices: list, positive_index_set: set) -> tuple:
    """Find the reciprocal rank of the first relevant item in a ranking.

    Args:
        ranked_indices (list): Passage indices ordered from most to least
            relevant according to the model.
        positive_index_set (set): Indices that count as relevant (positives).

    Returns:
        tuple: A (reciprocal_rank, first_positive_rank) pair. The first
            element is 1.0 / rank (float) for the first positive, and the
            second is its 1-based rank (int). Returns (0.0, None) when no
            positive is found.
    """
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in positive_index_set:
            return 1.0 / rank, rank
    return 0.0, None


def mrr_at_k_from_ranking(ranked_indices: list, positive_index_set: set, k: int) -> float:
    """Compute the reciprocal rank within the top-k of a single ranking.

    Args:
        ranked_indices (list): Passage indices ordered from most to least
            relevant according to the model.
        positive_index_set (set): Indices that count as relevant (positives).
        k (int): Cutoff rank for the evaluation.

    Returns:
        float: 1.0 / rank for the first positive within the top-k, or
            0.0 if no positive appears in the top-k.
    """
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in positive_index_set:
            return 1.0 / rank
    return 0.0


def build_pmcid_pools(data: list) -> dict:
    """Group passages into a per-PMCID candidate pool for local ranking.

    For each PMCID, positive and negative passages from every example are
    collected and de-duplicated, so each document is ranked against only its
    own pool of passages.

    Args:
        data (list): Grouped examples, each a dict with a PMCID key and
            optional positives and negatives lists of passage
            strings.

    Returns:
        dict: Mapping from PMCID to a pool dict with keys texts (the
            de-duplicated passage list) and text_to_index (a mapping from
            each passage string to its position in texts).
    """
    pmcid_to_texts = defaultdict(list)

    for ex in data:
        pmcid = ex["PMCID"]
        pmcid_to_texts[pmcid].extend(ex.get("positives", []))
        pmcid_to_texts[pmcid].extend(ex.get("negatives", []))

    pools = {}
    for pmcid, texts in pmcid_to_texts.items():
        uniq_texts = dedup_preserve_order(texts)
        pools[pmcid] = {
            "texts": uniq_texts,
            "text_to_index": {text: i for i, text in enumerate(uniq_texts)}
        }

    return pools


# Evaluation
def evaluate_pmcid_local(data: list) -> tuple:
    """Evaluate PMCID-local retrieval and aggregate ranking metrics.

    For each example the query is embedded and scored against its own PMCID
    pool, then Recall, Precision, NDCG, MAP and MRR are accumulated at each
    cutoff in K_VALUES. Examples without usable positives are skipped, and
    per-query top-TOP_K_SAVE rankings are collected for inspection.

    Args:
        data (list): Grouped examples, each a dict with PMCID,
            query and optional positives/negatives keys.

    Returns:
        tuple: A 10-element tuple
            (recall_at_k, precision_at_k, ndcg_at_k, map_at_k, mrr_at_k,
            mrr, first_positive_rank_distribution, all_rankings, skipped, n)
            where the first five are dicts keyed by k (float averages), mrr
            is the overall mean reciprocal rank (float),
            first_positive_rank_distribution is a Counter of first-positive
            ranks, all_rankings is a list of per-query ranking dicts,
            skipped (int) is the number of skipped examples, and n is the 
            number of evaluated queries.

    Raises:
        ValueError: If no valid queries remain after skipping.
    """
    pools = build_pmcid_pools(data)
    print(f"Built {len(pools)} PMCID-local pools.")

    pool_embeddings = {}
    for pmcid, pool_obj in tqdm(pools.items(), desc="Embedding PMCID pools"):
        texts = pool_obj["texts"]
        emb = embed_texts(texts, batch_size=BATCH_SIZE_PASSAGES)
        pool_embeddings[pmcid] = emb.to(DEVICE)

    recall_at_k = {k: 0.0 for k in K_VALUES}
    precision_at_k = {k: 0.0 for k in K_VALUES}
    ndcg_at_k = {k: 0.0 for k in K_VALUES}
    map_at_k = {k: 0.0 for k in K_VALUES}
    mrr_at_k = {k: 0.0 for k in K_VALUES}
    mrr_total = 0.0

    first_positive_rank_distribution = Counter()
    all_rankings = []
    skipped = 0

    for ex in tqdm(data, desc=f"Evaluating PMCID-local ranking {MODEL_NAME}"):
        pmcid = ex["PMCID"]
        query = ex["query"]
        positives = dedup_preserve_order(ex.get("positives", []))

        if not positives:
            skipped += 1
            continue

        texts = pools[pmcid]["texts"]
        text_to_index = pools[pmcid]["text_to_index"]
        passage_embeddings = pool_embeddings[pmcid]

        positive_indices = [text_to_index[p] for p in positives if p in text_to_index]
        positive_indices = dedup_preserve_order(positive_indices)
        positive_index_set = set(positive_indices)

        if not positive_index_set:
            skipped += 1
            continue

        query_embedding = embed_texts([query], batch_size=BATCH_SIZE_QUERIES).to(DEVICE)
        scores = torch.matmul(query_embedding, passage_embeddings.T).squeeze(0)

        ranked_indices = torch.argsort(scores, descending=True).tolist()

        rr, first_positive_rank = reciprocal_rank(ranked_indices, positive_index_set)
        mrr_total += rr
        if first_positive_rank is not None:
            first_positive_rank_distribution[first_positive_rank] += 1

        for k in K_VALUES:
            top_k = ranked_indices[:k]
            hits = sum(1 for idx in top_k if idx in positive_index_set)

            recall_at_k[k] += hits / len(positive_index_set)
            precision_at_k[k] += hits / k
            ndcg_at_k[k] += ndcg_at_k_from_ranking(ranked_indices, positive_index_set, k)
            map_at_k[k] += average_precision_at_k(ranked_indices, positive_index_set, k)
            mrr_at_k[k] += mrr_at_k_from_ranking(ranked_indices, positive_index_set, k)

        top_idxs, top_vals = topk_from_scores(scores, TOP_K_SAVE)

        top_ranking = []
        for rank, (idx, val) in enumerate(zip(top_idxs, top_vals), start=1):
            label = "POS" if idx in positive_index_set else "NEG"
            top_ranking.append({
                "rank": rank,
                "label": label,
                "score": float(val),
                "text": texts[idx],
            })

        positive_ranks = [
            rank for rank, idx in enumerate(ranked_indices, start=1)
            if idx in positive_index_set
        ]

        all_rankings.append({
            "PMCID": pmcid,
            "query": query,
            "num_positives": len(positive_index_set),
            "positive_ranks": positive_ranks,
            "first_positive_rank": first_positive_rank,
            "pool_size": len(texts),
            f"top_{TOP_K_SAVE}_ranking": top_ranking
        })

    n = len(data) - skipped
    if n <= 0:
        raise ValueError("No valid queries were evaluated.")

    recall_at_k = {k: v / n for k, v in recall_at_k.items()}
    precision_at_k = {k: v / n for k, v in precision_at_k.items()}
    ndcg_at_k = {k: v / n for k, v in ndcg_at_k.items()}
    map_at_k = {k: v / n for k, v in map_at_k.items()}
    mrr_at_k = {k: v / n for k, v in mrr_at_k.items()}
    mrr = mrr_total / n

    return (
        recall_at_k,
        precision_at_k,
        ndcg_at_k,
        map_at_k,
        mrr_at_k,
        mrr,
        first_positive_rank_distribution,
        all_rankings,
        skipped,
        n
    )


if __name__ == "__main__":
    data = load_jsonl(JSON_FILE)
    print(f"Total grouped examples: {len(data)}")

    (
        recall_at_k,
        precision_at_k,
        ndcg_at_k,
        map_at_k,
        mrr_at_k,
        mrr,
        rank_dist,
        all_rankings,
        skipped,
        evaluated
    ) = evaluate_pmcid_local(data)

    #with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    #    json.dump(all_rankings, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print(f"RESULTS – {MODEL_NAME} (PMCID LOCAL)")
    print("==============================")

    print(f"\nEvaluated queries: {evaluated}")
    print(f"Skipped queries:   {skipped}")

    print("\nRecall@K:")
    for k, v in recall_at_k.items():
        print(f"Recall@{k}: {v:.5f}")

    print("\nPrecision@K:")
    for k, v in precision_at_k.items():
        print(f"Precision@{k}: {v:.5f}")

    print("\nNDCG@K:")
    for k, v in ndcg_at_k.items():
        print(f"NDCG@{k}: {v:.5f}")

    print("\nMAP@K:")
    for k, v in map_at_k.items():
        print(f"MAP@{k}: {v:.5f}")

    print("\nMRR@K:")
    for k, v in mrr_at_k.items():
        print(f"MRR@{k}: {v:.5f}")

    print(f"\nMRR: {mrr:.5f}")

    print("\nFirst positive rank distribution (sample):")
    for rank in sorted(rank_dist)[:20]:
        print(f"Rank {rank}: {rank_dist[rank]}")
    if len(rank_dist) > 20:
        print("...")

    #print(f"\nComplete rankings (TOP-{TOP_K_SAVE}) saved in: {OUTPUT_JSON}")
