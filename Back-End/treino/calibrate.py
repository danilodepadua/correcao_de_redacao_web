"""Calibra os cortes que transformam a previsão contínua do modelo em notas 0, 40, ..., 200.

O arredondamento padrão usa cortes fixos (20, 60, 100, 140, 180). Como o modelo
treinado com MSE comprime as previsões para o meio, notas 0, 40 e 200 quase não
aparecem. Este script ajusta os 5 cortes de cada competência no conjunto de
VALIDAÇÃO para maximizar o QWK e grava o resultado em saved_model/config.json,
que a API usa na inferência. O conjunto de teste é usado só para medir o efeito.

Uso (dentro de Back-End):  python -m treino.calibrate [--model-dir saved_model] [--tema curto]
"""
import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # permite rodar também como "python treino/calibrate.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from avaliador.caminhos import PASTA_MODELO  # noqa: E402
from avaliador.dados import TIPOS_DE_TEMA, load_splits  # noqa: E402
from avaliador.modelo import MAX_LEN, MODEL_NAME, EnemDataset, carregar_modelo, ler_config, predict  # noqa: E402
from avaliador.notas import DEFAULT_CUTS, apply_cuts, fit_cuts, qwk  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Calibra os cortes das notas na validação.")
    ap.add_argument("--model-dir", default=str(PASTA_MODELO))
    ap.add_argument("--tema", default="curto", choices=TIPOS_DE_TEMA,
                    help="tema usado para ajustar e medir os cortes (padrão: curto, como no site)")
    args = ap.parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    config = ler_config(args.model_dir)
    max_len = int(config.get("max_len", MAX_LEN))
    divisao = config.get("divisao", "aleatoria")  # a mesma usada no treino

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = carregar_modelo(args.model_dir, device)

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
