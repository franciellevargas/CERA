import json
from collections import defaultdict

INPUT_FILE = "contriever_test_triplets_SUBJ_LOCAL_b.jsonl"

OUTPUT_LOCAL = "test_local_grouped.jsonl"
OUTPUT_GLOBAL = "test_global_grouped.jsonl"


def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of records.

    Args:
        path (str): Path to the JSONL file to read.

    Returns:
        list: One parsed dict per non-empty line in the file.
    """
    rows = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def group_by_pmcid_query(rows: list) -> dict:
    """Group triplet rows by their (PMCID, query) pair.

    Args:
        rows (list): Triplet records, each a dict with the keys PMCID,
            query, positive and negatives.

    Returns:
        dict: Maps each (pmcid, query) tuple to a dict with accumulated
            positives and negatives lists.
    """
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


def dedup_preserve_order(items: list) -> list:
    """Remove duplicate items while keeping their first-seen order.

    Args:
        items (list): Items to deduplicate.

    Returns:
        list: The unique items in the order they first appeared.
    """
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def build_local_dataset(groups: dict) -> list:
    """Build the local evaluation dataset from grouped triplets.

    Deduplicates the positives and negatives of each group and drops any
    negative that also appears as a positive.

    Args:
        groups (dict): Mapping of (pmcid, query) tuples to dicts holding
            positives and negatives lists, as produced by
            group_by_pmcid_query.

    Returns:
        list: One dict per group with the keys PMCID, query,
            positives and negatives.
    """
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


def save_jsonl(data: list, path: str):
    """Write a list of records to a JSONL file, one record per line.

    Args:
        data (list): Records (dicts) to serialise.
        path (str): Destination path for the JSONL file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rows = load_jsonl(INPUT_FILE)
    groups = group_by_pmcid_query(rows)

    local_dataset = build_local_dataset(groups)

    save_jsonl(local_dataset, OUTPUT_LOCAL)

    print(f"Original rows: {len(rows)}")
    print(f"Groups (PMCID, query): {len(groups)}")
    print(f"Local file saved to: {OUTPUT_LOCAL}")
