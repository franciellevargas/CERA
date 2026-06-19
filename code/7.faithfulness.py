import json
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# =========================================================
# GPU
# =========================================================

def get_free_gpu():
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=memory.free',
                '--format=csv,nounits,noheader'
            ],
            encoding='utf-8'
        )

        mem_free = [int(x) for x in result.strip().split('\n')]
        gpu_index = mem_free.index(max(mem_free))

        return gpu_index

    except Exception:
        return None


gpu_id = get_free_gpu()

DEVICE = torch.device(
    f'cuda:{gpu_id}'
) if gpu_id is not None and torch.cuda.is_available() else torch.device('cpu')

print(f"Using device: {DEVICE}")


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "triplet+kl_maskpos51_1e-6_8_sched_lambda0p1_epoch4"
JSON_FILE = "contriever_test_triplets_SUBJ_LOCAL_dedup.jsonl"

MAX_LENGTH = 128
BATCH_SIZE = 8

METHOD = "ratio"
TOP_K = 20
RATIO = 0.20
MIN_K = 5

USE_PAD_MASKING = True

SEED = 51
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# DATASET
# =========================================================

class TripletDataset1N(Dataset):

    def __init__(self, json_file, tokenizer, max_length=128):

        self.data = []

        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                if item.get("query") and item.get("positive") and item.get("negatives"):
                    if len(item["negatives"]) > 0:
                        self.data.append(item)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        query_enc = self.tokenizer(
            item["query"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        pos_enc = self.tokenizer(
            item["positive"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "query": query_enc["input_ids"].squeeze(0),
            "query_attn": query_enc["attention_mask"].squeeze(0),
            "positive": pos_enc["input_ids"].squeeze(0),
            "positive_attn": pos_enc["attention_mask"].squeeze(0)
        }


# =========================================================
# METRICS
# =========================================================

class RetrievalFaithfulnessMetrics:

    def __init__(self, model, tokenizer, device,
                 top_k=20, ratio=0.2, min_k=5, use_pad_masking=True):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.top_k = top_k
        self.ratio = ratio
        self.min_k = min_k
        self.use_pad_masking = use_pad_masking

    def encode_cls(self, input_ids, attention_mask):

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.unsqueeze(0).to(self.device),
                return_dict=True
            )

            emb = outputs.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=1)

        return emb.squeeze(0)

    def cosine_sim(self, emb1, emb2):
        return F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

    # =====================================================
    # FIXED MASKING
    # =====================================================

    def create_masked_input(self, input_ids, attention_mask, indices_to_mask):

        masked_ids = input_ids.clone()
        masked_attn = attention_mask.clone()

        for idx in indices_to_mask:

            if idx < 0 or idx >= len(masked_ids):
                continue

            token = self.tokenizer.convert_ids_to_tokens(
                [masked_ids[idx].item()]
            )[0]

            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if self.use_pad_masking:
                masked_ids[idx] = self.tokenizer.pad_token_id
                masked_attn[idx] = 0  
            else:
                masked_ids[idx] = self.tokenizer.mask_token_id

        return masked_ids, masked_attn

    # =====================================================

    def select_rationale_indices(self, attention_weights, attention_mask, method='ratio'):

        valid_attention = attention_weights.copy()
        valid_attention[attention_mask == 0] = -np.inf

        valid_attention[0] = -np.inf  # CLS

        seq_len = int(attention_mask.sum())
        valid_attention[seq_len - 1] = -np.inf  # SEP

        valid_count = seq_len - 2

        if method == 'ratio':
            k = max(self.min_k, int(valid_count * self.ratio))
        else:
            k = min(self.top_k, valid_count)

        indices = valid_attention.argsort()[-k:][::-1]

        return np.array([i for i in indices if valid_attention[i] > -np.inf])

    # =====================================================
    # COMPREHENSIVENESS
    # =====================================================

    def compute_comprehensiveness(self, query_ids, query_attn,
                                  doc_ids, doc_attn, rationale_indices):

        query_emb = self.encode_cls(query_ids, query_attn)
        doc_emb = self.encode_cls(doc_ids, doc_attn)

        original_sim = self.cosine_sim(query_emb, doc_emb)

        masked_ids, masked_attn = self.create_masked_input(
            doc_ids, doc_attn, rationale_indices
        )

        masked_emb = self.encode_cls(masked_ids, masked_attn)

        masked_sim = self.cosine_sim(query_emb, masked_emb)

        return original_sim - masked_sim

    # =====================================================
    # SUFFICIENCY (FIXED)
    # =====================================================

    def compute_sufficiency(self, query_ids, query_attn,
                            doc_ids, doc_attn, rationale_indices):

        query_emb = self.encode_cls(query_ids, query_attn)

        rationale_set = set(rationale_indices.tolist())

        indices_to_mask = [
            idx for idx in range(len(doc_ids))
            if idx not in rationale_set
        ]

        rationale_ids, rationale_attn = self.create_masked_input(
            doc_ids, doc_attn, indices_to_mask
        )

        rationale_emb = self.encode_cls(rationale_ids, rationale_attn)

        return self.cosine_sim(query_emb, rationale_emb)

    # =====================================================

    def compute_all(self, dataloader, attentions):

        comp_scores = []
        suff_scores = []

        idx = 0

        for batch in tqdm(dataloader):

            for i in range(batch["query"].size(0)):

                if idx >= len(attentions):
                    break

                query_ids = batch["query"][i]
                query_attn = batch["query_attn"][i]

                doc_ids = batch["positive"][i]
                doc_attn = batch["positive_attn"][i]

                attn = attentions[idx]

                rationale_indices = self.select_rationale_indices(
                    attn,
                    doc_attn.numpy()
                )

                if len(rationale_indices) == 0:
                    idx += 1
                    continue

                comp_scores.append(
                    self.compute_comprehensiveness(
                        query_ids, query_attn,
                        doc_ids, doc_attn,
                        rationale_indices
                    )
                )

                suff_scores.append(
                    self.compute_sufficiency(
                        query_ids, query_attn,
                        doc_ids, doc_attn,
                        rationale_indices
                    )
                )

                idx += 1

        return {
            "comprehensiveness": float(np.mean(comp_scores)),
            "sufficiency": float(np.mean(suff_scores))
        }


# =========================================================
# LOAD MODEL
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager"
).to(DEVICE)
model.eval()


# =========================================================
# DATA
# =========================================================

dataset = TripletDataset1N(JSON_FILE, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# ATTENTIONS
# =========================================================

all_attentions = []

with torch.no_grad():
    for batch in tqdm(dataloader):

        outputs = model(
            input_ids=batch["positive"].to(DEVICE),
            attention_mask=batch["positive_attn"].to(DEVICE),
            output_attentions=True
        )

        attn = outputs.attentions[-1][:, :, 0, :].mean(dim=1)

        all_attentions.extend(attn.cpu().numpy())


# =========================================================
# RUN
# =========================================================

metrics = RetrievalFaithfulnessMetrics(
    model,
    tokenizer,
    DEVICE,
    top_k=TOP_K,
    ratio=RATIO,
    min_k=MIN_K,
    use_pad_masking=USE_PAD_MASKING
)

results = metrics.compute_all(dataloader, all_attentions)

print("\n==============================")
print("FAITHFULNESS RESULTS")
print("==============================")
print(f"Comprehensiveness: {results['comprehensiveness']:.6f}")
print(f"Sufficiency: {results['sufficiency']:.6f}")