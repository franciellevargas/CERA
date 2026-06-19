import json
from collections import defaultdict

INPUT_FILE = "contriever_test_triplets_SUBJ_LOCAL_b.jsonl"

OUTPUT_LOCAL = "test_local_grouped.jsonl"
OUTPUT_GLOBAL = "test_global_grouped.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_pmcid_query(rows):
    groups = defaultdict(lambda: {"positives": [], "negatives": []})

    for row in rows:
        pmcid = row["PMCID"]
        query = row["query"]
        positive = row["positive"]
        negatives = row["negatives"]

        key = (pmcid, query)

        groups[key]["positives"].append(positive)
        groups[key]["negatives"].extend(negatives)

    return groups


def dedup_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_local_dataset(groups):
    local_dataset = []

    for (pmcid, query), data in groups.items():
        positives = dedup_preserve_order(data["positives"])
        negatives = dedup_preserve_order(data["negatives"])

        positive_set = set(positives)
        negatives = [n for n in negatives if n not in positive_set]

        local_dataset.append({
            "PMCID": pmcid,
            "query": query,
            "positives": positives,
            "negatives": negatives
        })

    return local_dataset


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rows = load_jsonl(INPUT_FILE)
    groups = group_by_pmcid_query(rows)

    local_dataset = build_local_dataset(groups)

    save_jsonl(local_dataset, OUTPUT_LOCAL)

    print(f"Linhas originais: {len(rows)}")
    print(f"Grupos (PMCID, query): {len(groups)}")
    print(f"Arquivo local salvo em: {OUTPUT_LOCAL}")
