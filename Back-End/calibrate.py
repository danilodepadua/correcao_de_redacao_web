"""Calibra os cortes que transformam a previsão contínua do modelo em notas 0, 40, ..., 200.

O arredondamento padrão usa cortes fixos (20, 60, 100, 140, 180). Como o modelo
treinado com MSE comprime as previsões para o meio, notas 0, 40 e 200 quase não
aparecem. Este script ajusta os 5 cortes de cada competência no conjunto de
VALIDAÇÃO para maximizar o QWK e grava o resultado em saved_model/config.json,
que a API usa na inferência. O conjunto de teste é usado só para medir o efeito.

Uso:  python calibrate.py [--model-dir ./saved_model]
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from train import (
    MAX_LEN,
    MODEL_NAME,
    EnemDataset,
    EnemEssayEvaluator,
    load_splits,
)

DEFAULT_CUTS = [20.0, 60.0, 100.0, 140.0, 180.0]
NOTAS = np.array([0, 40, 80, 120, 160, 200])


def apply_cuts(preds_0_200: np.ndarray, cuts) -> np.ndarray:
    """Converte previsões contínuas (escala 0–200) em notas, dados 5 cortes crescentes."""
    return NOTAS[np.searchsorted(np.asarray(cuts), preds_0_200, side="right")]


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def fit_cuts(preds: np.ndarray, targets: np.ndarray, rounds: int = 3) -> list:
    """Busca coordenada a coordenada: ajusta um corte por vez mantendo a ordem."""
    cuts = list(DEFAULT_CUTS)
    best = qwk(targets, apply_cuts(preds, cuts))
    for _ in range(rounds):
        for i in range(5):
            lo = cuts[i - 1] + 1 if i > 0 else -40.0
            hi = cuts[i + 1] - 1 if i < 4 else 240.0
            for c in np.arange(lo, hi + 0.5, 1.0):
                trial = cuts[:i] + [float(c)] + cuts[i + 1:]
                score = qwk(targets, apply_cuts(preds, trial))
                if score > best + 1e-9:
                    best, cuts = score, trial
    return [round(c, 1) for c in cuts]


def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.append(out.float().cpu().numpy())
            targets.append(batch["labels"].numpy())
    return np.vstack(preds) * 200.0, np.round(np.vstack(targets) * 200.0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./saved_model")
    ap.add_argument("--tema", default="curto", choices=("trecho", "curto", "vazio", "misto"),
                    help="tema usado para ajustar e medir os cortes (padrão: curto, como no site)")
    args = ap.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    max_len = int(config.get("max_len", MAX_LEN))
    divisao = config.get("divisao", "aleatoria")  # a mesma usada no treino

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = EnemEssayEvaluator(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model_weights.pt"),
                                     map_location=device, weights_only=True))

    splits = load_splits(tema=args.tema, divisao=divisao)
    loaders = {
        nome: DataLoader(EnemDataset(txt, temas, y, tokenizer, max_len), batch_size=16)
        for nome, (temas, txt, y) in splits.items() if nome in ("val", "test")
    }
    val_p, val_y = predict(model, loaders["val"], device)
    test_p, test_y = predict(model, loaders["test"], device)

    cuts = [fit_cuts(val_p[:, i], val_y[:, i]) for i in range(5)]

    def report(nome, p, y):
        antes = [qwk(y[:, i], apply_cuts(p[:, i], DEFAULT_CUTS)) for i in range(5)]
        depois = [qwk(y[:, i], apply_cuts(p[:, i], cuts[i])) for i in range(5)]
        print(f"{nome}: QWK antes {np.mean(antes):.4f} {[round(a, 3) for a in antes]}")
        print(f"{nome}: QWK depois {np.mean(depois):.4f} {[round(d, 3) for d in depois]}")
        return float(np.mean(antes)), float(np.mean(depois)), depois

    print("Cortes por competência:", cuts)
    report("Validação", val_p, val_y)
    t_antes, t_depois, t_comp = report("Teste", test_p, test_y)

    config.update({"max_len": max_len, "model_name": MODEL_NAME, "cuts": cuts, "tema_calibracao": args.tema})
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results_path = os.path.join(args.model_dir, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    results["calibration"] = {
        "cuts": cuts,
        "test_qwk_sem_calibracao": round(t_antes, 4),
        "test_qwk_com_calibracao": round(t_depois, 4),
        "test_qwk_por_competencia_com_calibracao": [round(c, 4) for c in t_comp],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Cortes gravados em {config_path}")


if __name__ == "__main__":
    main()
