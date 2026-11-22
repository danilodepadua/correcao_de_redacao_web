"""Experimento de viés: o modelo premia características superficiais?

Aplica mudanças controladas às redações de validação e teste (que o modelo não
viu no treino) e mede como a nota prevista reage. Há dois grupos de mudanças:

- Superficiais (não deveriam aumentar a nota): inserir conectivos no início dos
  parágrafos, repetir um parágrafo, colar uma citação famosa sem relação com o tema.
- Que pioram o texto (deveriam diminuir a nota): tirar os acentos, embaralhar as
  frases, remover a conclusão.

Para não confundir o efeito com o corte de tokens, cada mudança só é medida nas
redações em que o texto original e o alterado cabem inteiros em max_len.

Uso:  python analise_vies.py [--model-dir ./saved_model]
Saída: tabela no terminal e saved_model/analise_vies.json
"""
import argparse
import json
import os
import random
import re
import unicodedata

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train import MAX_LEN, MODEL_NAME, EnemDataset, EnemEssayEvaluator, load_splits

NOTAS = np.array([0, 40, 80, 120, 160, 200])
CONECTIVOS = ["Além disso, ", "Ademais, ", "Nesse sentido, ", "Outrossim, ", "Dessa forma, "]
CITACAO = (
    'Como afirmou o filósofo Aristóteles, "o homem é, por natureza, um animal político", '
    "o que reforça a relevância dessa discussão."
)


def paragrafos(texto):
    return [p.strip() for p in texto.split("\n") if p.strip()]


def frases(texto):
    return [f for f in re.split(r"(?<=[.!?])\s+", texto.replace("\n", " ")) if f.strip()]


def minusculo_inicio(p):
    return p[:1].lower() + p[1:] if p else p


# Cada mudança recebe o texto e devolve o texto alterado (ou None se não se aplica)
def conectivos(t):
    ps = paragrafos(t)
    if len(ps) < 3:
        return None
    novos = [ps[0]] + [CONECTIVOS[(i - 1) % len(CONECTIVOS)] + minusculo_inicio(p) for i, p in enumerate(ps[1:-1], 1)]
    return "\n".join(novos + ["Portanto, " + minusculo_inicio(ps[-1])])


def repetir_paragrafo(t):
    ps = paragrafos(t)
    if len(ps) < 3:
        return None
    return "\n".join(ps[:2] + [ps[1]] + ps[2:])


def citacao_generica(t):
    ps = paragrafos(t)
    if len(ps) < 2:
        return None
    return "\n".join([ps[0] + " " + CITACAO] + ps[1:])


def tirar_acentos(t):
    sem = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return sem if sem != t else None


def embaralhar_frases(t, rng=random.Random(0)):
    fs = frases(t)
    if len(fs) < 4:
        return None
    fs = fs[:]
    rng.shuffle(fs)
    return " ".join(fs)


def remover_conclusao(t):
    ps = paragrafos(t)
    if len(ps) < 3:
        return None
    return "\n".join(ps[:-1])


MUDANCAS = [
    ("Conectivos no início dos parágrafos", conectivos, "superficial"),
    ("Repetir um parágrafo inteiro", repetir_paragrafo, "superficial"),
    ("Citação famosa sem relação com o tema", citacao_generica, "superficial"),
    ("Tirar todos os acentos", tirar_acentos, "piora"),
    ("Embaralhar a ordem das frases", embaralhar_frases, "piora"),
    ("Remover a conclusão", remover_conclusao, "piora"),
]


def prever(model, tok, temas, textos, max_len, device):
    ds = EnemDataset(textos, temas, None, tok, max_len)
    saida = []
    model.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=16):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            saida.append(out.float().cpu().numpy())
    return np.vstack(saida) * 200.0


def converter(p, cuts):
    return np.stack([NOTAS[np.searchsorted(cuts[i], p[:, i], side="right")] for i in range(5)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./saved_model")
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    max_len = int(cfg.get("max_len", MAX_LEN))
    cuts = cfg.get("cuts") or [[20, 60, 100, 140, 180]] * 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    model = EnemEssayEvaluator(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "model_weights.pt"),
                                     map_location=device, weights_only=True))

    s = load_splits(tema="curto")  # título curto, como no site
    temas = list(s["val"][0]) + list(s["test"][0])
    textos = list(s["val"][1]) + list(s["test"][1])

    def cabe(tema, texto):
        return len(tok(f"Tema: {tema} [SEP] Redação: {texto}")["input_ids"]) <= max_len

    resultados = []
    print(f"Redações analisadas (validação + teste): {len(textos)} | max_len {max_len}\n")
    cab = f"{'Mudança':40} {'n':>4} {'Δ total':>8} {'C1':>6} {'C2':>6} {'C3':>6} {'C4':>6} {'C5':>6} {'nota subiu':>10}"
    print(cab)
    print("-" * len(cab))
    for nome, fn, grupo in MUDANCAS:
        pares = [(tm, tx, fn(tx)) for tm, tx in zip(temas, textos)]
        pares = [(tm, tx, nv) for tm, tx, nv in pares if nv and cabe(tm, tx) and cabe(tm, nv)]
        tm = [p[0] for p in pares]
        antes = prever(model, tok, tm, [p[1] for p in pares], max_len, device)
        depois = prever(model, tok, tm, [p[2] for p in pares], max_len, device)
        delta = (depois - antes).mean(0)  # escala contínua 0–200 por competência
        final_antes = converter(antes, cuts).sum(1)
        final_depois = converter(depois, cuts).sum(1)
        subiu = float((final_depois > final_antes).mean())
        caiu = float((final_depois < final_antes).mean())
        r = {
            "mudanca": nome, "grupo": grupo, "n": len(pares),
            "delta_total": round(float(delta.sum()), 1),
            "delta_por_competencia": [round(float(d), 1) for d in delta],
            "nota_final_subiu": round(subiu, 3), "nota_final_caiu": round(caiu, 3),
        }
        resultados.append(r)
        print(f"{nome:40} {len(pares):>4} {delta.sum():>+8.1f} "
              + " ".join(f"{d:>+6.1f}" for d in delta) + f" {100 * subiu:>9.0f}%")

    out = os.path.join(args.model_dir, "analise_vies.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nΔ = variação média da nota contínua (escala 0–200 por competência; total 0–1000).")
    print(f"Resultados gravados em {out}")


if __name__ == "__main__":
    main()
