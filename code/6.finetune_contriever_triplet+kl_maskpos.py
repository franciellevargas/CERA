import json
import numpy as np
import os
import random
import subprocess
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim

from tqdm import tqdm


# -----------------------------
# Function to select the GPU with the most available memory
# -----------------------------
def get_free_gpu():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        mem_free = [int(x) for x in result.strip().split('\n')]
        gpu_index = mem_free.index(max(mem_free))
        return gpu_index
    except Exception as e:
        print("It was not possible to access nvidia-smi:", e)
        return None


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "facebook/contriever"
BATCH_SIZE = 8
MAX_LENGTH = 128
LR = 1e-5
EPOCHS = 30
MARGIN = 0.2

# KL loss weight
LAMBDA_RAT = 0.05
EPS = 1e-8

JSON_FILE = "contriever_train_triplets_SUBJ_LOCAL_dedup.jsonl"
SAVE_DIR = "final-contriever-finetuned-triplet+kl_maskpos51"

REPORT_FILE = f"{SAVE_DIR}_training_report.txt"

# Create or overwrite the report file and write the header
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(f"TRAINING REPORT - {SAVE_DIR}\n")
    f.write(f"MODEL_NAME={MODEL_NAME} | LR={LR} | BATCH_SIZE={BATCH_SIZE} | MAX_LENGTH={MAX_LENGTH} | LAMBDA_RAT={LAMBDA_RAT}\n")
    f.write("="*60 + "\n")


# Checkpoints per epoch (1-indexed)
CHECKPOINT_EPOCHS = {1, 5, 10, 15, 20, 25}

gpu_id = get_free_gpu()
DEVICE = torch.device(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")


# -----------------------------
# Seed setup for reproducibility
# -----------------------------
SEED = 51

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------
# Triplet 1-N dataset with rationale mask (full positive mask)
# -----------------------------
class TripletDataset1N(Dataset):
    def __init__(self, json_file, tokenizer, max_length=MAX_LENGTH):
        self.data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                query = item.get("query")
                pos = item.get("positive")
                neg_list = item.get("negatives")
                if query and pos and neg_list and len(neg_list) > 0:
                    self.data.append({"query": query, "positive": pos, "negatives": neg_list})
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

        neg_enc_list = [
            self.tokenizer(
                neg,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            ) for neg in item["negatives"]
        ]

        # Rationale mask: full positive mask
        # 1 for valid tokens (attention_mask == 1), 0 for PAD
        # CLS token (position 0) is removed to avoid forcing attention on CLS
        rat_mask = pos_enc["attention_mask"].squeeze(0).float()
        rat_mask[0] = 0.0

        return {
            "query": query_enc["input_ids"].squeeze(0),
            "query_attn": query_enc["attention_mask"].squeeze(0),

            "positive": pos_enc["input_ids"].squeeze(0),
            "positive_attn": pos_enc["attention_mask"].squeeze(0),

            "rat_mask": rat_mask,

            "negatives": torch.stack([neg["input_ids"].squeeze(0) for neg in neg_enc_list]),
            "negatives_attn": torch.stack([neg["attention_mask"].squeeze(0) for neg in neg_enc_list])
        }


# -----------------------------
# Model: CLS embedding plus CLS-to-token attention (last layer)
# -----------------------------
class ContrieverFineTuneWithAttn(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True
        )
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def encode_positive_with_attention(self, input_ids, attention_mask):
        """
        Returns:
          - Normalized CLS embedding [B, dim]
          - CLS-to-token attention averaged across heads [B, T]
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)

        # Last layer attention: [B, H, T, T]
        attn_last = out.attentions[-1]
        # CLS query index is 0, CLS-to-token attention: [B, H, T]
        cls_to_tok = attn_last[:, :, 0, :]
        # Average across heads: [B, T]
        cls_to_tok = cls_to_tok.mean(dim=1)

        return emb, cls_to_tok


# -----------------------------
# Cosine Triplet Loss 1-N
# -----------------------------
class CosineTripletLoss1N(nn.Module):
    def __init__(self, margin=0.2, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negatives):
        # anchor: [B, dim], positive: [B, dim], negatives: [B, N, dim]
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(1)
        sim_neg = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)
        losses = F.relu(sim_neg - sim_pos + self.margin)
        losses = losses.mean(dim=1)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


# -----------------------------
# KL(r || a) for rationale alignment (positives only)
# -----------------------------
def rationale_kl_loss(cls_to_tok, rat_mask, pos_attn, eps=1e-8):
    """
    cls_to_tok: [B, T] CLS-to-token attention
    rat_mask:   [B, T] binary rationale mask with CLS removed
    pos_attn:   [B, T] attention mask of the positive
    """
    # Valid tokens: non-PAD and non-CLS
    valid = (pos_attn.float() * (rat_mask > 0).float())

    # Attention distribution a: mask and renormalize
    a = cls_to_tok * valid
    a = a / a.sum(dim=1, keepdim=True).clamp(min=eps)
    a = a.clamp(min=eps)

    # Target distribution r: uniform over valid rationale tokens
    r = rat_mask * pos_attn.float()
    r = r / r.sum(dim=1, keepdim=True).clamp(min=eps)
    r = r.clamp(min=eps)

    # KL divergence KL(r || a)
    kl = (r * (r.log() - a.log())).sum(dim=1)
    return kl.mean()


# -----------------------------
# Tokenizer and DataLoader
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = TripletDataset1N(JSON_FILE, tokenizer, max_length=MAX_LENGTH)
g = torch.Generator()
g.manual_seed(SEED)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g
)


# -----------------------------
# Model initialization
# -----------------------------
model = ContrieverFineTuneWithAttn(MODEL_NAME).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
triplet_loss = CosineTripletLoss1N(margin=MARGIN)


# -----------------------------
# Training loop: Triplet + KL + Total loss
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_trip = 0.0
    total_kl = 0.0
    total_all = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()

        # Query embeddings
        query_ids = batch["query"].to(DEVICE)
        query_attn = batch["query_attn"].to(DEVICE)
        query_emb = model.encode_cls(query_ids, query_attn)

        # Positive embeddings and attention
        pos_ids = batch["positive"].to(DEVICE)
        pos_attn = batch["positive_attn"].to(DEVICE)
        rat_mask = batch["rat_mask"].to(DEVICE)

        pos_emb, cls_to_tok = model.encode_positive_with_attention(pos_ids, pos_attn)

        # Negatives flattening
        B, N, L = batch["negatives"].shape
        neg_input_ids = batch["negatives"].view(B * N, L).to(DEVICE)
        neg_attention_mask = batch["negatives_attn"].view(B * N, L).to(DEVICE)
        neg_emb = model.encode_cls(neg_input_ids, neg_attention_mask).view(B, N, -1)

        # Triplet loss
        loss_trip = triplet_loss(query_emb, pos_emb, neg_emb)

        # KL rationale loss
        loss_kl = rationale_kl_loss(cls_to_tok, rat_mask, pos_attn, eps=EPS)

        # Total loss
        loss = loss_trip + (LAMBDA_RAT * loss_kl)

        loss.backward()
        optimizer.step()

        total_trip += float(loss_trip.item())
        total_kl += float(loss_kl.item())
        total_all += float(loss.item())

    denom = max(1, len(dataloader))
    log_line = (
        f"Epoch {epoch+1} | "
        f"Triplet: {total_trip/denom:.4f} | "
        f"KL: {total_kl/denom:.4f} | "
        f"Total: {total_all/denom:.4f}\n"
    )

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

    # -----------------------------
    # Save checkpoints at selected epochs
    # -----------------------------
    epoch_num = epoch + 1
    if epoch_num in CHECKPOINT_EPOCHS:
        ckpt_dir = f"{SAVE_DIR}_epoch{epoch_num}"
        model.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved at: {ckpt_dir}")


# -----------------------------
# Save final fine-tuned model
# -----------------------------
FINAL_SAVE_DIR = f"{SAVE_DIR}_epoch{epoch_num}"
model.encoder.save_pretrained(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"Fine-tuning completed. Model saved at: {FINAL_SAVE_DIR}")
