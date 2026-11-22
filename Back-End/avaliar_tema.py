"""Mede como o tipo de tema na entrada afeta o modelo (QWK no conjunto de teste).

No treino original, o "tema" era um trecho de até 300 caracteres do texto
motivador da proposta; no site, o usuário digita um título curto. Este script
avalia o mesmo modelo com cada tipo de tema (trecho, título curto, vazio),
usando os cortes calibrados de saved_model/config.json.

Uso:  python avaliar_tema.py [--model-dir ./saved_model]
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from calibrate import DEFAULT_CUTS, apply_cuts, predict, qwk
from train import MAX_LEN, MODEL_NAME, EnemDataset, EnemEssayEvaluator, load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./saved_model")
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    max_len = int(cfg.get("max_len", MAX_LEN))
    cuts = cfg.get("cuts") or [DEFAULT_CUTS] * 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = EnemEssayEvaluator(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model_weights.pt"),
                                     map_location=device, weights_only=True))

    resultados = {}
    for tema in ("trecho", "curto", "vazio"):
        temas, textos, y = load_splits(tema=tema)["test"]
        loader = DataLoader(EnemDataset(textos, temas, y, tok, max_len), batch_size=16)
        p, yt = predict(model, loader, device)
        por_comp = [qwk(yt[:, i], apply_cuts(p[:, i], cuts[i])) for i in range(5)]
        resultados[tema] = {"qwk": round(float(np.mean(por_comp)), 4),
                            "qwk_por_competencia": [round(float(q), 4) for q in por_comp]}
        print(f"tema {tema:7} QWK teste {np.mean(por_comp):.4f} {[round(q, 3) for q in por_comp]}")

    with open(os.path.join(args.model_dir, "avaliacao_tema.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
