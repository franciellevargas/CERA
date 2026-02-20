import json
import hashlib

INPUT_JSONL = "contriever_train_triplets_ev_offset.jsonl"
OUTPUT_JSONL = f"{INPUT_JSONL}_dedup.jsonl"


def normalize_obj(obj):
    """
    Normalizes a JSON object for comparison:
    - strips leading/trailing spaces from strings
    - preserves list order (negatives), because this may matter for training
    """
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_obj(x) for x in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def stable_fingerprint(obj):
    """
    Generates a stable signature for the JSON (independent of key order).
    """
    norm = normalize_obj(obj)
    payload = json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dedup_jsonl_by_key(input_path, output_path):
    """
    Removes duplicates using the logical key: (PMCID, query, positive, negatives).
    Useful if extra fields vary between entries.
    """
    seen = set()
    kept = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] line {line_num}: invalid JSON, skipping.")
                continue

            pmcid = (obj.get("PMCID") or "").strip()
            query = (obj.get("query") or "").strip()
            positive = (obj.get("positive") or "").strip()
            negatives = obj.get("negatives") or []
            negatives = [n.strip() if isinstance(n, str) else str(n) for n in negatives]

            key_obj = {"PMCID": pmcid, "query": query, "positive": positive, "negatives": negatives}
            fp = stable_fingerprint(key_obj)

            if fp in seen:
                dropped += 1
                continue

            seen.add(fp)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("   Dedup (by key) completed.")
    print(f"   Kept:     {kept}")
    print(f"   Removed:  {dropped}")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    dedup_jsonl_by_key(INPUT_JSONL, OUTPUT_JSONL)
