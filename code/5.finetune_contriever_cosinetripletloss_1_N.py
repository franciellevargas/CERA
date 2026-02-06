import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import subprocess

# -----------------------------
# Função para GPU
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
        print("Não foi possível acessar nvidia-smi:", e)
        return None

# -----------------------------
# Configurações
# -----------------------------
MODEL_NAME = "facebook/contriever"
BATCH_SIZE = 8  
MAX_LENGTH = 128
LR = 1e-5
EPOCHS = 15
JSON_FILE = "contriever_train_triplets_chunk50_SAFE.jsonl"
SAVE_DIR = "FINAL_contriever-finetuned-cosinetripletloss-1_5_subj"

gpu_id = get_free_gpu()
DEVICE = torch.device(f'cuda:{gpu_id}') if gpu_id is not None and torch.cuda.is_available() else torch.device('cpu')
print(f"Usando dispositivo: {DEVICE}")

# -----------------------------
# Dataset Triplet 1-N
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
        query_enc = self.tokenizer(item["query"], truncation=True, padding="max_length",
                                   max_length=self.max_length, return_tensors="pt")
        pos_enc = self.tokenizer(item["positive"], truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
        neg_enc_list = [self.tokenizer(neg, truncation=True, padding="max_length",
                                       max_length=self.max_length, return_tensors="pt") for neg in item["negatives"]]
        return {
            "query": query_enc["input_ids"].squeeze(0),
            "query_attn": query_enc["attention_mask"].squeeze(0),
            "positive": pos_enc["input_ids"].squeeze(0),
            "positive_attn": pos_enc["attention_mask"].squeeze(0),
            "negatives": torch.stack([neg["input_ids"].squeeze(0) for neg in neg_enc_list]),
            "negatives_attn": torch.stack([neg["attention_mask"].squeeze(0) for neg in neg_enc_list])
        }

# -----------------------------
# Modelo com Mean Pooling
# -----------------------------
class ContrieverFineTune(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        embeddings = torch.sum(last_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

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
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(1)  # [B,1]
        sim_neg = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)  # [B,N]
        losses = F.relu(sim_neg - sim_pos + self.margin)  # [B,N]
        losses = losses.mean(dim=1)  # média sobre negativos
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

# -----------------------------
# Tokenizer e DataLoader
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = TripletDataset1N(JSON_FILE, tokenizer, max_length=MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Inicialização do modelo
# -----------------------------
model = ContrieverFineTune(MODEL_NAME).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
triplet_loss = CosineTripletLoss1N(margin=0.2)

# -----------------------------
# Loop de Treino 1-N
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        # Embeddings
        query_emb = model(batch["query"].to(DEVICE), batch["query_attn"].to(DEVICE))
        pos_emb = model(batch["positive"].to(DEVICE), batch["positive_attn"].to(DEVICE))
        # Achata negativos
        B, N, L = batch["negatives"].shape
        neg_input_ids = batch["negatives"].view(B*N, L).to(DEVICE)
        neg_attention_mask = batch["negatives_attn"].view(B*N, L).to(DEVICE)
        neg_emb = model(neg_input_ids, neg_attention_mask).view(B, N, -1)
        # Loss
        loss = triplet_loss(query_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# Salvar modelo fine-tuned
# -----------------------------
model.encoder.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Fine-tuning concluído e modelo salvo em:", SAVE_DIR)
