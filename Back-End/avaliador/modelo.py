"""Arquitetura do avaliador, dataset de entrada e predição em lote."""
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Tamanho máximo padrão da sequência (tema + redação) em tokens (limite do BERT).
# O valor usado em cada treino fica salvo em saved_model/config.json,
# e a API (main.py) lê de lá para usar exatamente o mesmo na inferência.
MAX_LEN = 512


class EnemEssayEvaluator(nn.Module):
    """BERTimbau + dropout + camada linear com 5 saídas (uma por competência, escala 0–1)."""

    def __init__(self, model_name=MODEL_NAME, dropout_rate=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(cls_output))


class EnemDataset(Dataset):
    """Redações no formato "Tema: <tema> [SEP] Redação: <texto>"; notas na escala 0–1."""

    def __init__(self, texts, themes, scores, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.themes = themes
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        combined_text = f"Tema: {self.themes[idx]} [SEP] Redação: {self.texts[idx]}"
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.scores is not None:
            # Escala normalizada [0, 1]
            item["labels"] = torch.tensor(self.scores[idx] / 200.0, dtype=torch.float)
        return item


def predict(model, loader, device):
    """Previsões contínuas (escala 0–200) e notas verdadeiras (inteiras) de um DataLoader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.append(out.float().cpu().numpy())
            if "labels" in batch:
                targets.append(batch["labels"].numpy())
    p = np.vstack(preds) * 200.0
    y = np.round(np.vstack(targets) * 200.0).astype(int) if targets else None
    return p, y


def ler_config(pasta_modelo) -> dict:
    """config.json do modelo (max_len, cortes, tipo de tema, divisão...), ou {} se não existir."""
    caminho = os.path.join(pasta_modelo, "config.json")
    if not os.path.exists(caminho):
        return {}
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def carregar_modelo(pasta_modelo, device):
    """Carrega EnemEssayEvaluator com os pesos de pasta_modelo, em modo de avaliação."""
    model = EnemEssayEvaluator(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(os.path.join(pasta_modelo, "model_weights.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    return model
