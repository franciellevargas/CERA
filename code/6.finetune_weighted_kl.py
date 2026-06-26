import json
import numpy as np
import random
import subprocess
import spacy
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch import nn, optim

from tqdm import tqdm


def get_free_gpu():
    """Find the GPU with the most free memory by querying nvidia-smi.

    Returns:
        int | None: Index of the GPU with the largest amount of free memory,
            or None if nvidia-smi could not be accessed.
    """
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


MODEL_NAME = "facebook/contriever"
BATCH_SIZE = 8
MAX_LENGTH = 128
LR = 1e-6
EPOCHS = 10
MARGIN = 0.2

LAMBDA_RAT = 0.05
EPS = 1e-8

JSON_FILE = "contriever_train_triplets_ev_offset_dedup.jsonl"
SAVE_DIR = "triplet+kl_weighted_b_1e-6_8_sched_lambda0p05"

REPORT_FILE = f"{SAVE_DIR}_training_report.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(f"TRAINING REPORT - {SAVE_DIR}\n")
    f.write(
        f"MODEL_NAME={MODEL_NAME} | LR={LR} | BATCH_SIZE={BATCH_SIZE} | "
        f"MAX_LENGTH={MAX_LENGTH} | LAMBDA_RAT={LAMBDA_RAT}\n"
    )
    f.write("=" * 60 + "\n")

CHECKPOINT_EPOCHS = {1,2,3,4,5,6,7,8,9}

gpu_id = get_free_gpu()
DEVICE = torch.device(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

SEED = 51

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

nlp = spacy.load("en_core_web_sm")

POS_WEIGHTS = {
    "NOUN": 1.0,

    "PROPN": 0.3,
    "VERB": 0.1,
    "ADJ": 0.1,
    "ADV": 0.0,

    "NUM": 0.0,
    "PRON": 0.0,
    "DET": 0.0,
    "ADP": 0.0,
    "AUX": 0.0,
    "CCONJ": 0.0,
    "SCONJ": 0.0,
    "PART": 0.0,
    "PUNCT": 0.0,
    "SPACE": 0.0,
    "X": 0.0
}


class TripletDataset1N(Dataset):
    """Dataset of (query, positive, negatives) triplets with weighted rationales.

    Each line of the source JSONL file is expected to contain a query, a single
    positive passage, a list of negatives and an evidence span (character
    offsets) inside the positive. Entries missing any of these fields (or with
    an invalid evidence span) are skipped. At load time each positive is parsed
    with spaCy so that, at access time, a part-of-speech weighted rationale mask
    can be built over the evidence tokens of the positive passage.
    """

    def __init__(self, json_file: str, tokenizer, max_length: int = MAX_LENGTH):
        """Load and filter the triplets, caching spaCy token spans per positive.

        Args:
            json_file (str): Path to the JSONL file holding the triplets.
            tokenizer: Hugging Face tokenizer used to encode the texts.
            max_length (int): Maximum sequence length used for truncation and
                padding when tokenizing.
        """
        self.data = []

        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                query = item.get("query")
                pos = item.get("positive")
                neg_list = item.get("negatives")

                ev_s = item.get("evidence_start")
                ev_e = item.get("evidence_end")

                if (
                    query and
                    pos and
                    neg_list and
                    len(neg_list) > 0 and
                    ev_s is not None and
                    ev_e is not None and
                    int(ev_e) > int(ev_s) >= 0
                ):

                    # spaCy preprocessing
                    doc = nlp(pos)

                    spacy_tokens = []

                    for token in doc:
                        spacy_tokens.append({
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "pos": token.pos_
                        })

                    self.data.append({
                        "query": query,
                        "positive": pos,
                        "negatives": neg_list,
                        "evidence_start": int(ev_s),
                        "evidence_end": int(ev_e),
                        "spacy_tokens": spacy_tokens
                    })

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of triplets in the dataset.

        Returns:
            int: The number of valid triplets loaded.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Tokenize the triplet and build the weighted rationale mask.

        For the positive passage, tokens overlapping the evidence span are
        assigned a weight from POS_WEIGHTS based on the (majority-voted)
        part-of-speech of the overlapping spaCy tokens. The CLS position is
        zeroed out, and if the evidence vanished after truncation the mask falls
        back to the (CLS-zeroed) attention mask.

        Args:
            idx (int): Index of the triplet to retrieve.

        Returns:
            dict: A mapping with the tokenized tensors for the query, positive,
                and negatives, namely query, query_attn, positive,
                positive_attn, rat_mask (weighted rationale mask over the
                positive with the CLS position zeroed out), negatives and
                negatives_attn.
        """
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
            return_offsets_mapping=True,
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

        ev_s = int(item["evidence_start"])
        ev_e = int(item["evidence_end"])

        attn = pos_enc["attention_mask"].squeeze(0)
        offsets = pos_enc["offset_mapping"].squeeze(0)

        rat_mask = torch.zeros_like(attn, dtype=torch.float)

        spacy_tokens = item["spacy_tokens"]

        # Build rationale mask
        for i in range(attn.size(0)):

            if attn[i].item() == 0:
                continue

            ts = int(offsets[i, 0].item())
            te = int(offsets[i, 1].item())

            # Skip special tokens
            if ts == 0 and te == 0:
                continue

            # Token must intersect evidence
            if te <= ev_s or ts >= ev_e:
                continue

            # Find overlapping spaCy tokens
            overlapping_pos = []

            for sp_tok in spacy_tokens:

                sp_s = sp_tok["start"]
                sp_e = sp_tok["end"]

                overlap = not (te <= sp_s or ts >= sp_e)

                if overlap:
                    overlapping_pos.append(sp_tok["pos"])

            # Fallback
            if len(overlapping_pos) == 0:
                pos_tag = "X"

            else:
                # Majority voting
                pos_tag = max(
                    set(overlapping_pos),
                    key=overlapping_pos.count
                )

            weight = POS_WEIGHTS.get(pos_tag, 0.5)

            rat_mask[i] = weight

        # Remove CLS
        rat_mask[0] = 0.0

        # Fallback if evidence vanished after truncation
        if rat_mask.sum().item() == 0:
            rat_mask = attn.float()
            rat_mask[0] = 0.0

        return {
            "query": query_enc["input_ids"].squeeze(0),
            "query_attn": query_enc["attention_mask"].squeeze(0),

            "positive": pos_enc["input_ids"].squeeze(0),
            "positive_attn": attn,

            "rat_mask": rat_mask,

            "negatives": torch.stack([
                neg["input_ids"].squeeze(0)
                for neg in neg_enc_list
            ]),

            "negatives_attn": torch.stack([
                neg["attention_mask"].squeeze(0)
                for neg in neg_enc_list
            ])
        }


class ContrieverFineTuneWithAttn(nn.Module):
    """Contriever encoder wrapper that exposes CLS embeddings and attention.

    Wraps a Hugging Face AutoModel (loaded with eager attention so that
    attention weights are available) and provides helpers to produce
    L2-normalized CLS embeddings as well as the CLS-to-token attention used by
    the rationale KL loss.
    """

    def __init__(self, model_name: str):
        """Build the wrapper around a pretrained encoder.

        Args:
            model_name (str): Hugging Face model identifier or path to load the
                encoder from.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")

    def encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch and return L2-normalized CLS embeddings.

        Args:
            input_ids (torch.Tensor): Token ids of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape
                (batch, seq_len).

        Returns:
            torch.Tensor: L2-normalized CLS embeddings of shape
                (batch, hidden_size).
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True
        )
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def encode_positive_with_attention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """Encode a batch and return CLS embeddings plus CLS-to-token attention.

        The CLS-to-token attention from the last layer is averaged across heads
        and is later compared against the rationale mask in the KL loss.

        Args:
            input_ids (torch.Tensor): Token ids of shape (batch, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape
                (batch, seq_len).

        Returns:
            tuple: A pair (emb, cls_to_tok) where emb is the L2-normalized
                CLS embedding of shape (batch, hidden_size) and cls_to_tok
                is the last-layer, head-averaged CLS-to-token attention of shape
                (batch, seq_len).
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)

        attn_last = out.attentions[-1]
        cls_to_tok = attn_last[:, :, 0, :]
        cls_to_tok = cls_to_tok.mean(dim=1)

        return emb, cls_to_tok


class CosineTripletLoss1N(nn.Module):
    """Cosine-similarity triplet margin loss with one positive and N negatives.

    For each anchor the loss encourages the cosine similarity to the positive to
    exceed the similarity to every negative by at least margin. The per-anchor
    loss is the mean over its negatives.
    """

    def __init__(self, margin: float = 0.2, reduction: str = 'mean'):
        """Configure the triplet loss.

        Args:
            margin (float): Margin enforced between the positive and negative
                similarities.
            reduction (str): How to reduce the per-anchor losses; one of
                mean, sum or any other value to return the unreduced per-anchor
                losses.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """Compute the triplet loss for a batch.

        Args:
            anchor (torch.Tensor): Anchor embeddings of shape
                (batch, hidden_size).
            positive (torch.Tensor): Positive embeddings of shape
                (batch, hidden_size).
            negatives (torch.Tensor): Negative embeddings of shape
                (batch, num_negatives, hidden_size).

        Returns:
            torch.Tensor: The reduced loss (a scalar for mean/sum) or the
                per-anchor losses of shape (batch,) otherwise.
        """
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


def rationale_kl_loss(cls_to_tok: torch.Tensor, rat_mask: torch.Tensor, pos_attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL divergence between the rationale mask and the CLS-to-token attention.

    Both the model's CLS-to-token attention and the rationale mask are restricted
    to valid (non-padding) rationale tokens and normalized into distributions
    over the sequence. The KL divergence KL(rationale || attention) is then
    averaged over the batch, encouraging the model to attend to rationale tokens.

    Args:
        cls_to_tok (torch.Tensor): CLS-to-token attention of shape
            (batch, seq_len).
        rat_mask (torch.Tensor): Weighted rationale mask of shape (batch, seq_len)
            marking the rationale tokens of the positive passage.
        pos_attn (torch.Tensor): Attention mask of the positive passage of shape
            (batch, seq_len), used to exclude padding tokens.
        eps (float): Small constant for numerical stability when normalizing and
            taking logarithms.

    Returns:
        torch.Tensor: Scalar mean KL divergence over the batch.
    """
    valid = (pos_attn.float() * (rat_mask > 0).float())

    a = cls_to_tok * valid
    a = a / a.sum(dim=1, keepdim=True).clamp(min=eps)
    a = a.clamp(min=eps)

    r = rat_mask * pos_attn.float()
    r = r / r.sum(dim=1, keepdim=True).clamp(min=eps)
    r = r.clamp(min=eps)

    kl = (r * (r.log() - a.log())).sum(dim=1)
    return kl.mean()


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

model = ContrieverFineTuneWithAttn(MODEL_NAME).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)

triplet_loss = CosineTripletLoss1N(margin=MARGIN)

total_training_steps = len(dataloader) * EPOCHS
num_warmup_steps = int(0.1 * total_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_training_steps
)

print(f"Total training steps: {total_training_steps}")
print(f"Warmup steps: {num_warmup_steps}")

for epoch in range(EPOCHS):
    model.train()
    total_trip = 0.0
    total_kl = 0.0
    total_all = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()

        query_ids = batch["query"].to(DEVICE)
        query_attn = batch["query_attn"].to(DEVICE)
        query_emb = model.encode_cls(query_ids, query_attn)

        pos_ids = batch["positive"].to(DEVICE)
        pos_attn = batch["positive_attn"].to(DEVICE)
        rat_mask = batch["rat_mask"].to(DEVICE)

        pos_emb, cls_to_tok = model.encode_positive_with_attention(pos_ids, pos_attn)

        B, N, L = batch["negatives"].shape
        neg_input_ids = batch["negatives"].view(B * N, L).to(DEVICE)
        neg_attention_mask = batch["negatives_attn"].view(B * N, L).to(DEVICE)
        neg_emb = model.encode_cls(neg_input_ids, neg_attention_mask).view(B, N, -1)

        loss_trip = triplet_loss(query_emb, pos_emb, neg_emb)
        loss_kl = rationale_kl_loss(cls_to_tok, rat_mask, pos_attn, eps=EPS)
        loss = loss_trip + (LAMBDA_RAT * loss_kl)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_trip += float(loss_trip.item())
        total_kl += float(loss_kl.item())
        total_all += float(loss.item())

    denom = max(1, len(dataloader))
    current_lr = scheduler.get_last_lr()[0]

    log_line = (
        f"Epoch {epoch+1} | "
        f"LR: {current_lr:.8f} | "
        f"Triplet: {total_trip/denom:.4f} | "
        f"KL: {total_kl/denom:.4f} | "
        f"Total: {total_all/denom:.4f}\n"
    )

    print(log_line.strip())

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)

    epoch_num = epoch + 1
    if epoch_num in CHECKPOINT_EPOCHS:
        ckpt_dir = f"{SAVE_DIR}_epoch{epoch_num}"
        model.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved at: {ckpt_dir}")


FINAL_SAVE_DIR = f"{SAVE_DIR}_epoch{epoch_num}"
model.encoder.save_pretrained(FINAL_SAVE_DIR)
tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"Fine-tuning completed. Model saved at: {FINAL_SAVE_DIR}")
