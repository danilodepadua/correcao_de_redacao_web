"""Leitura do Essay-BR e divisão em treino, validação e teste."""
import ast
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from avaliador.caminhos import ARQUIVO_PROPOSTAS, ARQUIVO_REDACOES, ARQUIVO_TEMAS_CURTOS

logger = logging.getLogger(__name__)

TIPOS_DE_TEMA = ("trecho", "curto", "vazio", "misto")
TIPOS_DE_DIVISAO = ("aleatoria", "por_tema")


def extract_prompt_text(val):
    """Resume o texto de uma proposta (prompts.csv) em até 300 caracteres."""
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
    # Normaliza espaços e limita a 300 caracteres para sobrar espaço para a redação
    return " ".join(s.split())[:300]


def load_splits(seed: int = 42, tema: str = "trecho", divisao: str = "aleatoria"):
    """Carrega o Essay-BR e devolve {"train"|"val"|"test": (temas, textos, notas)}.

    Divisão fixa 80/10/10 (random_state=42), a mesma em todos os experimentos,
    qualquer que seja o tipo de tema (só o texto do tema muda, não as redações).

    tema:
      "trecho" — até 300 caracteres do texto motivador da proposta (prompts.csv)
      "curto"  — título curto do tema (temas_curtos.csv), como o usuário digita no site
      "vazio"  — sem tema
      "misto"  — sorteia, para cada redação, trecho ou título curto (50% cada)

    divisao:
      "aleatoria" — sorteia redações (80/10/10); os mesmos temas aparecem nos três conjuntos
      "por_tema"  — sorteia temas: os temas da validação e do teste não aparecem no
                    treino (cerca de 80/10/10 das redações), para medir o desempenho
                    em temas inéditos
    """
    if divisao not in TIPOS_DE_DIVISAO:
        raise ValueError(f"divisao deve ser uma de {TIPOS_DE_DIVISAO}")
    if tema not in TIPOS_DE_TEMA:
        raise ValueError(f"tema deve ser um de {TIPOS_DE_TEMA}")
    # Carregamento e junção das redações (4570) com os temas (86)
    df = pd.read_csv(ARQUIVO_REDACOES)
    prompts_df = pd.read_csv(ARQUIVO_PROPOSTAS)

    prompts_df["tema_prompt"] = prompts_df["description"].apply(extract_prompt_text)
    df = df.merge(
        prompts_df[["id", "tema_prompt"]], left_on="prompt", right_on="id", how="left"
    )

    # Conversão das colunas serializadas para texto contínuo e matriz (4570, 5)
    X_texts = df["essay"].apply(lambda x: "\n".join(ast.literal_eval(x))).values
    trechos = df["tema_prompt"].fillna("").astype(str).values
    if tema == "trecho":
        X_themes = trechos
    elif tema == "vazio":
        X_themes = np.array([""] * len(df), dtype=object)
    else:
        curtos_df = pd.read_csv(ARQUIVO_TEMAS_CURTOS).set_index("id")["tema_curto"]
        curtos = df["prompt"].map(curtos_df).fillna("").astype(str).values
        if tema == "curto":
            X_themes = curtos
        else:  # misto: sorteio fixo, para o treino ver os dois tipos de tema
            usa_curto = np.random.default_rng(seed).random(len(df)) < 0.5
            X_themes = np.where(usa_curto, curtos, trechos)
    y = np.array(df["competence"].apply(ast.literal_eval).tolist(), dtype=np.float32)

    if divisao == "por_tema":
        # Sorteia temas, não redações: 20% dos temas saem do treino e se dividem
        # entre validação e teste (mesma semente, divisão fixa)
        grupos = df["prompt"].values
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        idx_tr, idx_tmp = next(g1.split(X_texts, groups=grupos))
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        sub_val, sub_te = next(g2.split(idx_tmp, groups=grupos[idx_tmp]))
        idx_val, idx_te = idx_tmp[sub_val], idx_tmp[sub_te]
        X_train_t, X_train_txt, y_train = X_themes[idx_tr], X_texts[idx_tr], y[idx_tr]
        X_val_t, X_val_txt, y_val = X_themes[idx_val], X_texts[idx_val], y[idx_val]
        X_test_t, X_test_txt, y_test = X_themes[idx_te], X_texts[idx_te], y[idx_te]
    else:
        # Divisão: 80% Treino, 10% Validação, 10% Teste (fixa, para comparar experimentos)
        X_train_t, X_temp_t, X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
            X_themes, X_texts, y, test_size=0.20, random_state=seed
        )
        X_val_t, X_test_t, X_val_txt, X_test_txt, y_val, y_test = train_test_split(
            X_temp_t, X_temp_txt, y_temp, test_size=0.50, random_state=seed
        )
    return {
        "train": (X_train_t, X_train_txt, y_train),
        "val": (X_val_t, X_val_txt, y_val),
        "test": (X_test_t, X_test_txt, y_test),
    }
