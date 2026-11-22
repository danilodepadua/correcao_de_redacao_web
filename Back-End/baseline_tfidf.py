"""Modelo de referência clássico: TF-IDF + regressão linear (Ridge).

Serve para medir quanto o BERTimbau ganha sobre um modelo simples. Usa as mesmas
condições do modelo principal:
- mesma divisão de dados (load_splits) e mesmo tipo de tema no treino (--tema-treino);
- regularização escolhida pela validação;
- cortes das notas calibrados na validação com título curto (como no site);
- QWK no teste com título curto (uso real) e com trecho do texto motivador.

Uso:  python baseline_tfidf.py [--tema-treino misto]
Saída: tabela no terminal e baseline_tfidf.json
"""
import argparse
import json

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from calibrate import DEFAULT_CUTS, apply_cuts, fit_cuts, qwk
from train import TIPOS_DE_TEMA, load_splits


def juntar(parte):
    temas, textos, _ = parte
    return [f"{t} \n {x}" for t, x in zip(temas, textos)]


def media_qwk(y, p, cuts):
    por = [qwk(y[:, i], apply_cuts(p[:, i], cuts[i])) for i in range(5)]
    return float(np.mean(por)), [round(float(v), 4) for v in por]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tema-treino", choices=TIPOS_DE_TEMA, default="misto")
    args = ap.parse_args()

    treino = load_splits(tema=args.tema_treino)
    curto, trecho = load_splits(tema="curto"), load_splits(tema="trecho")
    y_tr = treino["train"][2]
    y_val = curto["val"][2].astype(int)
    y_te = curto["test"][2].astype(int)

    # Palavras (1 e 2 palavras seguidas) + pedaços de caracteres (pegam erros de grafia)
    palavras = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    caracteres = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=100000, sublinear_tf=True)
    textos_tr = juntar(treino["train"])
    X_tr = hstack([palavras.fit_transform(textos_tr), caracteres.fit_transform(textos_tr)]).tocsr()

    def vetor(parte):
        t = juntar(parte)
        return hstack([palavras.transform(t), caracteres.transform(t)]).tocsr()

    X_val_treino = vetor(treino["val"])
    X_val_curto, X_te_curto, X_te_trecho = vetor(curto["val"]), vetor(curto["test"]), vetor(trecho["test"])

    # Escolhe a regularização pela validação (mesmo tipo de tema do treino, como no BERTimbau)
    melhor = None
    for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        m = Ridge(alpha=alpha).fit(X_tr, y_tr)
        q, _ = media_qwk(treino["val"][2].astype(int), m.predict(X_val_treino), [DEFAULT_CUTS] * 5)
        print(f"alpha={alpha:<5} QWK validação (arredondamento simples) {q:.4f}")
        if melhor is None or q > melhor[0]:
            melhor = (q, alpha, m)
    _, alpha, modelo = melhor

    p_val = modelo.predict(X_val_curto)
    cuts = [fit_cuts(p_val[:, i], y_val[:, i]) for i in range(5)]
    p_curto, p_trecho = modelo.predict(X_te_curto), modelo.predict(X_te_trecho)

    resultados = {
        "modelo": "TF-IDF (palavras 1-2 + caracteres 2-5) + Ridge",
        "tema_treino": args.tema_treino,
        "alpha": alpha,
        "atributos": int(X_tr.shape[1]),
    }
    print(f"\nTF-IDF + Ridge (alpha={alpha}, tema no treino: {args.tema_treino}) | {X_tr.shape[1]} atributos")
    for nome, p in (("curto", p_curto), ("trecho", p_trecho)):
        simples, simples_c = media_qwk(y_te, p, [DEFAULT_CUTS] * 5)
        calib, calib_c = media_qwk(y_te, p, cuts)
        resultados[f"teste_tema_{nome}"] = {
            "qwk_simples": round(simples, 4), "qwk_calibrado": round(calib, 4),
            "qwk_calibrado_por_competencia": calib_c,
        }
        print(f"teste, tema {nome:6}: QWK arredondamento simples {simples:.4f} | cortes calibrados {calib:.4f} {calib_c}")

    with open("baseline_tfidf.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
