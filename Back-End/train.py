import ast
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Tamanho máximo da sequência (tema + redação) em tokens.
# Deve ser o mesmo valor usado na inferência (MAX_LEN em main.py).
MAX_LEN = 384


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class EnemEssayEvaluator(nn.Module):
    def __init__(
        self, model_name="neuralmind/bert-base-portuguese-cased", dropout_rate=0.2
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(cls_output))


class EnemDataset(Dataset):
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


def post_process_scores(preds_array):
    raw_scores = preds_array * 200.0
    clipped = np.clip(raw_scores, 0, 200)
    rounded = (np.round(clipped / 40.0) * 40).astype(int)
    return rounded


def extract_prompt_text(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # IDs 0 a 42 estão serializados como lista de parágrafos
    if s.startswith("[") and s.endswith("]"):
        try:
            paragraphs = ast.literal_eval(s)
            if isinstance(paragraphs, list) and paragraphs:
                keywords = ("tema", "dissertação", "redija", "ponto de vista")
                cmd = next(
                    (p for p in paragraphs if any(k in p.lower() for k in keywords)),
                    None,
                )
                s = cmd if cmd else " ".join(str(p) for p in paragraphs)
        except (SyntaxError, ValueError) as exc:
            logger.warning("Não foi possivel serializar o prompt: %s", exc)
    # Normaliza espaços e limita a 300 caracteres para sobrar espaço na janela de MAX_LEN tokens para a redação
    return " ".join(s.split())[:300]


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(ids, mask)
            total_loss += criterion(preds, labels).item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)

    y_pred_rounded = post_process_scores(preds_arr)
    y_true_rounded = np.round(targets_arr * 200.0).astype(int)

    qwk_scores = [
        cohen_kappa_score(
            y_true_rounded[:, i], y_pred_rounded[:, i], weights="quadratic"
        )
        for i in range(5)
    ]
    return avg_loss, qwk_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

    # Carregamento e junção das redações (4570) com os temas (86)
    df = pd.read_csv("essay-br.csv")
    prompts_df = pd.read_csv("prompts.csv")

    prompts_df["tema_prompt"] = prompts_df["description"].apply(extract_prompt_text)
    df = df.merge(
        prompts_df[["id", "tema_prompt"]], left_on="prompt", right_on="id", how="left"
    )

    # Conversão das colunas serializadas para texto contínuo e matriz (4570, 5)
    X_texts = df["essay"].apply(lambda x: "\n".join(ast.literal_eval(x))).values
    X_themes = df["tema_prompt"].fillna("").astype(str).values
    y = np.array(df["competence"].apply(ast.literal_eval).tolist(), dtype=np.float32)

    # Divisão: 80% Treino, 10% Validação, 10% Teste
    X_train_t, X_temp_t, X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
        X_themes, X_texts, y, test_size=0.20, random_state=42
    )
    X_val_t, X_test_t, X_val_txt, X_test_txt, y_val, y_test = train_test_split(
        X_temp_t, X_temp_txt, y_temp, test_size=0.50, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = EnemDataset(X_train_txt, X_train_t, y_train, tokenizer)
    val_dataset = EnemDataset(X_val_txt, X_val_t, y_val, tokenizer)
    test_dataset = EnemDataset(X_test_txt, X_test_t, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = EnemEssayEvaluator(model_name=MODEL_NAME).to(device)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:9]:
        for param in layer.parameters():
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Treinamento e salvamento do melhor checkpoint
    epochs = 3
    log_every = 10
    best_val_loss = float("inf")
    os.makedirs("./saved_model", exist_ok=True)

    total_batches = len(train_loader)
    train_start_time = time.perf_counter()
    epoch_durations = []

    print(
        f"Iniciando treinamento em: {device} | Treino: {len(train_dataset)} ({total_batches} batches) | "
        f"Val: {len(val_dataset)} | Teste: {len(test_dataset)}",
        flush=True,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.perf_counter()
        last_log_time = epoch_start

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(ids, mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            now = time.perf_counter()
            if (
                step % log_every == 0
                or step == total_batches
                or (now - last_log_time) >= 30
            ):
                last_log_time = now
                elapsed_epoch = now - epoch_start
                sec_per_batch = elapsed_epoch / step
                eta_epoch = sec_per_batch * (total_batches - step)
                avg_train_so_far = train_loss / step

                print(
                    f"  [Época {epoch + 1}/{epochs}] Batch {step}/{total_batches} "
                    f"({100.0 * step / total_batches:.1f}%) | "
                    f"Loss: {avg_train_so_far:.4f} | "
                    f"Decorrido: {format_duration(elapsed_epoch)} | "
                    f"ETA Época: {format_duration(eta_epoch)}",
                    flush=True,
                )

        val_start = time.perf_counter()
        avg_val_loss, qwk_scores = evaluate(model, val_loader, criterion, device)
        val_duration = time.perf_counter() - val_start

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)

        avg_epoch_sec = float(np.mean(epoch_durations))
        remaining_epochs = epochs - (epoch + 1)
        eta_total = avg_epoch_sec * remaining_epochs
        total_elapsed = time.perf_counter() - train_start_time

        print(
            f"Época {epoch + 1}/{epochs} concluída em {format_duration(epoch_duration)} "
            f"(val: {format_duration(val_duration)}) | "
            f"Train Loss: {train_loss / total_batches:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Tempo total: {format_duration(total_elapsed)} | "
            f"ETA restante: {format_duration(eta_total)}",
            flush=True,
        )
        print(
            f"QWK Médio Validação: {np.mean(qwk_scores):.4f} "
            f"(C1-C5: {[round(s, 4) for s in qwk_scores]})",
            flush=True,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./saved_model/model_weights.pt")
            tokenizer.save_pretrained("./saved_model/tokenizer")
            print("-> Novo melhor modelo salvo em ./saved_model", flush=True)

    # Avaliação do modelo
    model.load_state_dict(
        torch.load("./saved_model/model_weights.pt", map_location=device)
    )
    test_loss, test_qwk = evaluate(model, test_loader, criterion, device)
    total_time = time.perf_counter() - train_start_time
    print("\n--- RESULTADO FINAL NO CONJUNTO DE TESTE ---", flush=True)
    print(f"Tempo total de execução: {format_duration(total_time)}", flush=True)
    print(f"Test Loss (MSE): {test_loss:.4f}", flush=True)
    print(
        f"QWK Médio Teste: {np.mean(test_qwk):.4f} "
        f"(C1-C5: {[round(s, 4) for s in test_qwk]})",
        flush=True,
    )


if __name__ == "__main__":
    main()