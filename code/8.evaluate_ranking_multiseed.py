import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter, defaultdict
from statistics import mean, pstdev

# =============================
# CONFIG
# =============================
JSON_FILE = "contriever_test_triplets_SUBJ_LOCAL_dedup.jsonl"

MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS_TO_EVAL = [1, 2, 3, 5, 10]
K_VALUES = [1, 3, 5, 10, 50]

TECHNIQUES = ["evmask", "maskpos"]
SEEDS = [42, 51, 60]

NAME_PATTERNS = [
    "contriever-finetuned-triplet+kl_{tech}{seed}_epoch{epoch}",
]

RUNS_LOG_FILE = "eval_runs_log.txt"
FINAL_REPORT_FILE = "eval_final_report.txt"
SUMMARY_JSON = "eval_summary.json"


# =============================
# UTILS
# =============================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resolve_checkpoint_name(tech: str, seed: int, epoch: int) -> str | None:
    for pat in NAME_PATTERNS:
        name = pat.format(tech=tech, seed=seed, epoch=epoch)
        if os.path.isdir(name):
            return name
    return None


def write_block(f, lines, end="\n"):
    for ln in lines:
        f.write(ln + "\n")
    f.write(end)
    f.flush()

# =============================
# METRICS (per checkpoint)
# =============================
@torch.no_grad()
def embed(texts, tokenizer, model):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    output = model(**encoded)
    emb = output.last_hidden_state[:, 0]
    emb = F.normalize(emb, p=2, dim=1)
    return emb


def evaluate_checkpoint(model_name: str, data):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    recall_at_k = {k: 0 for k in K_VALUES}
    hit_at_1 = 0
    mrr_total = 0.0
    rank_distribution = Counter()

    for ex in tqdm(data, desc=f"🔎 {model_name}", leave=False):
        query = ex["query"]
        positive = ex["positive"]
        negatives = ex["negatives"]

        passages = [positive] + negatives
        labels = ["POS"] + ["NEG"] * len(negatives)

        q_emb = embed([query], tokenizer, model)
        p_emb = embed(passages, tokenizer, model)

        scores = torch.matmul(q_emb, p_emb.T).squeeze(0).tolist()

        ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)

        positive_rank = None
        for i, (label, _) in enumerate(ranked, start=1):
            if label == "POS":
                positive_rank = i
                break

        rank_distribution[positive_rank] += 1
        mrr_total += 1.0 / positive_rank

        if positive_rank == 1:
            hit_at_1 += 1

        for k in K_VALUES:
            if positive_rank <= k:
                recall_at_k[k] += 1

    n = len(data)
    recall_at_k = {k: v / n for k, v in recall_at_k.items()}
    hit_at_1 = hit_at_1 / n
    mrr = mrr_total / n

    return {
        "recall_at_k": recall_at_k,
        "hit_at_1": hit_at_1,
        "mrr": mrr,
        "rank_distribution": dict(rank_distribution),
    }


# =============================
# MAIN
# =============================
def main():
    data = load_jsonl(JSON_FILE)
    print(f"Total de exemplos: {len(data)}")

    open(RUNS_LOG_FILE, "w", encoding="utf-8").close()
    open(FINAL_REPORT_FILE, "w", encoding="utf-8").close()

    grouped = defaultdict(list)
    all_runs_struct = []

    with open(RUNS_LOG_FILE, "a", encoding="utf-8") as logf:
        write_block(logf, [
            "==============================",
            "EVAL RUNS LOG (one-by-one)",
            "==============================",
            f"JSON_FILE: {JSON_FILE}",
            f"EPOCHS_TO_EVAL: {EPOCHS_TO_EVAL}",
            f"TECHNIQUES: {TECHNIQUES}",
            f"SEEDS: {SEEDS}",
            f"DEVICE: {DEVICE}",
            ""
        ])

        for tech in TECHNIQUES:
            for seed in SEEDS:
                for epoch in EPOCHS_TO_EVAL:
                    ckpt = resolve_checkpoint_name(tech, seed, epoch)
                    if ckpt is None:
                        write_block(logf, [
                            "----------------------------------------",
                            f"CONFIG: tech={tech} seed={seed} epoch={epoch}",
                            "STATUS: SKIPPED (checkpoint not found)",
                            ""
                        ])
                        continue

                    write_block(logf, [
                        "----------------------------------------",
                        f"CONFIG: tech={tech} seed={seed} epoch={epoch}",
                        f"MODEL_NAME: {ckpt}",
                    ])

                    metrics = evaluate_checkpoint(ckpt, data)

                    lines = ["METRICS:"]
                    for k in K_VALUES:
                        lines.append(f"  Recall@{k}: {metrics['recall_at_k'][k]:.6f}")
                    lines.append(f"  Hit@1:    {metrics['hit_at_1']:.6f}")
                    lines.append(f"  MRR:      {metrics['mrr']:.6f}")
                    write_block(logf, lines + [""])

                    run_obj = {"tech": tech, "seed": seed, "epoch": epoch, "model_name": ckpt, **metrics}
                    all_runs_struct.append(run_obj)
                    grouped[(tech, epoch)].append(run_obj)

    summary = {}
    best_overall = None

    with open(FINAL_REPORT_FILE, "a", encoding="utf-8") as repf:
        repf.write("========================================\n")
        repf.write("FINAL REPORT (mean across seeds)\n")
        repf.write("========================================\n")
        repf.write(f"JSON_FILE: {JSON_FILE}\n")
        repf.write(f"EPOCHS_TO_EVAL: {EPOCHS_TO_EVAL}\n")
        repf.write(f"TECHNIQUES: {TECHNIQUES}\n")
        repf.write(f"SEEDS: {SEEDS}\n")
        repf.write("\n")

        for tech in TECHNIQUES:
            repf.write(f"========== Technique: {tech} ==========\n")
            for epoch in EPOCHS_TO_EVAL:
                runs = grouped.get((tech, epoch), [])
                if len(runs) == 0:
                    repf.write(f"- epoch {epoch}: NO RUNS FOUND\n")
                    continue

                mrrs = [r["mrr"] for r in runs]
                hits = [r["hit_at_1"] for r in runs]
                recall_lists = {k: [r["recall_at_k"][k] for r in runs] for k in K_VALUES}

                mean_mrr = mean(mrrs)
                mean_hit = mean(hits)
                mean_recalls = {k: mean(recall_lists[k]) for k in K_VALUES}
                std_mrr = pstdev(mrrs) if len(mrrs) > 1 else 0.0

                repf.write(f"\nEpoch {epoch} (n_seeds={len(runs)}):\n")
                for k in K_VALUES:
                    repf.write(f"  Recall@{k}: {mean_recalls[k]:.6f}\n")
                repf.write(f"  Hit@1:    {mean_hit:.6f}\n")
                repf.write(f"  MRR:      {mean_mrr:.6f} (std={std_mrr:.6f})\n")

                key = f"{tech}_epoch{epoch}"
                summary[key] = {
                    "tech": tech,
                    "epoch": epoch,
                    "n_seeds": len(runs),
                    "mean_recall_at_k": mean_recalls,
                    "mean_hit_at_1": mean_hit,
                    "mean_mrr": mean_mrr,
                    "std_mrr": std_mrr,
                    "seeds_present": sorted({r["seed"] for r in runs}),
                }

                candidate = (mean_mrr, tech, epoch)
                if best_overall is None or candidate[0] > best_overall[0]:
                    best_overall = candidate

            repf.write("\n")

        if best_overall is not None:
            repf.write("========================================\n")
            repf.write("BEST OVERALL (by mean MRR)\n")
            repf.write("========================================\n")
            repf.write(f"Technique: {best_overall[1]}\n")
            repf.write(f"Epoch:     {best_overall[2]}\n")
            repf.write(f"Mean MRR:  {best_overall[0]:.6f}\n")

    out = {
        "runs": all_runs_struct,
        "summary_mean_across_seeds": summary,
        "best_overall_by_mean_mrr": {
            "tech": best_overall[1],
            "epoch": best_overall[2],
            "mean_mrr": best_overall[0],
        } if best_overall else None
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f" Log incremental: {RUNS_LOG_FILE}")
    print(f" Report final:    {FINAL_REPORT_FILE}")
    print(f" Summary JSON:    {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
