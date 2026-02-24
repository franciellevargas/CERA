import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# --- Configurações ---
VAL_DATA_PATH = "contriever_valtest_candidates.jsonl"
MODEL_FINE = "contriever-finetuned-cosinetripletloss"
DEVICE = "cuda"
TOP_N = 3  # número de passagens a exibir

def load_candidates(path):
    queries, candidates = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            candidates.append(item["candidates"])  # lista de dicts {"label":..., "text":...}
    return queries, candidates

def explainable_ranking(model_name, queries, candidates, top_n=3):
    print(f"\nCarregando modelo: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)

    for q, cand_list in zip(queries, candidates):
        q_emb = model.encode(q)

        # calcula distância query ↔ candidato
        results = []
        for cand in cand_list:
            c_emb = model.encode(cand["text"])
            dist = cosine_distances([q_emb], [c_emb])[0][0]
            results.append((cand["label"], cand["text"], dist))

        # ordena por distância (quanto menor, mais relevante)
        results_sorted = sorted(results, key=lambda x: x[2])[:top_n]

        # --- Exibe ranking interpretável ---
        print(f"\n🔎 Query: {q}")
        for rank, (label, text, dist) in enumerate(results_sorted, start=1):
            print(f"  Rank {rank} | {label.upper():>8} | Distância: {dist:.4f}")
            print(f"    Evidência: {text}")

# --- Carrega dataset expandido ---
queries, candidates = load_candidates(VAL_DATA_PATH)

# --- Avaliação explicável (exemplo nas 3 primeiras queries) ---
explainable_ranking(MODEL_FINE, queries[:5], candidates[:5], top_n=TOP_N)
