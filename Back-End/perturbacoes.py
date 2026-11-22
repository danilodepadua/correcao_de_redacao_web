"""Aumento de dados contra os vieses do avaliador (usado por train.py --aumento-vies).

A análise de viés (analise_vies.py) mostrou que o modelo premia texto mais longo e
repertório genérico, e quase não percebe quando as ideias são embaralhadas. Para
corrigir isso no treino, acrescentamos ao conjunto de TREINO cópias alteradas de
parte das redações, com uma nota que não sobe (ou cai), seguindo a grade do ENEM:

  mudança                               nota da cópia alterada
  repetir um parágrafo                  C3 -40 e C4 -40 (repetição prejudica progressão e coesão)
  citação famosa sem relação com o tema igual à original (repertório não pertinente não pontua)
  embaralhar as frases                  C3 -80 e C4 -80 (texto sem organização nem encadeamento)

As penalidades são escolhas da equipe com base na grade, não notas de corretores.

Para a avaliação ser honesta, as versões daqui são DIFERENTES das usadas em
analise_vies.py: o parágrafo repetido é sorteado, as citações vêm de outra lista
(a citação de Aristóteles da análise não aparece aqui) e o embaralhamento usa outra
semente. Conectivos inseridos, acentos removidos e conclusão removida ficam fora
do treino, para testar se o efeito se generaliza a manipulações não vistas.
"""
import random
import re

import numpy as np

CITACOES_TREINO = [
    ("o filósofo Sócrates", "só sei que nada sei"),
    ("o filósofo René Descartes", "penso, logo existo"),
    ("o filósofo Friedrich Nietzsche", "aquilo que não me mata me fortalece"),
    ("o filósofo Jean-Paul Sartre", "o homem está condenado a ser livre"),
    ("o filósofo Heráclito", "ninguém pode entrar duas vezes no mesmo rio"),
    ("o filósofo Thomas Hobbes", "o homem é o lobo do homem"),
    ("o filósofo Platão", "a parte que ignoramos é muito maior que tudo quanto sabemos"),
    ("o escritor Machado de Assis", "o tempo é um rato roedor das coisas"),
]
MODELOS_CITACAO = [
    'Como afirmou {autor}, "{frase}", o que reforça a importância dessa discussão.',
    'Nesse contexto, vale lembrar {autor}, para quem "{frase}".',
    'Segundo {autor}, "{frase}", ideia que dialoga diretamente com o tema.',
]

# Índices das competências na matriz de notas
C3, C4 = 2, 3


def _paragrafos(texto):
    return [p.strip() for p in texto.split("\n") if p.strip()]


def _frases(texto):
    return [f for f in re.split(r"(?<=[.!?])\s+", texto.replace("\n", " ")) if f.strip()]


def repetir_paragrafo(texto, rng):
    ps = _paragrafos(texto)
    if len(ps) < 3:
        return None
    i = rng.randrange(1, len(ps) - 1)          # um parágrafo de desenvolvimento
    j = rng.randrange(i + 1, len(ps))          # repetido mais adiante no texto
    return "\n".join(ps[:j] + [ps[i]] + ps[j:])


def citacao_irrelevante(texto, rng):
    ps = _paragrafos(texto)
    if len(ps) < 2:
        return None
    autor, frase = rng.choice(CITACOES_TREINO)
    citacao = rng.choice(MODELOS_CITACAO).format(autor=autor, frase=frase)
    i = rng.randrange(0, len(ps) - 1)
    ps = ps[:]
    ps[i] = ps[i] + " " + citacao
    return "\n".join(ps)


def embaralhar_frases(texto, rng):
    fs = _frases(texto)
    if len(fs) < 4:
        return None
    fs = fs[:]
    rng.shuffle(fs)
    return " ".join(fs)


# (nome, função, variação da nota por competência)
AUMENTOS = [
    ("repetir_paragrafo", repetir_paragrafo, {C3: -40, C4: -40}),
    ("citacao_irrelevante", citacao_irrelevante, {}),
    ("embaralhar_frases", embaralhar_frases, {C3: -80, C4: -80}),
]


def aumentar_contra_vies(temas, textos, notas, fracao, seed=42):
    """Devolve (temas, textos, notas) com cópias alteradas de uma fração das redações.

    Cada redação sorteada (com probabilidade `fracao`) ganha uma cópia com uma das
    mudanças de AUMENTOS, também sorteada. As notas são clipadas em [0, 200].
    """
    rng = random.Random(seed + 1000)
    novos_temas, novos_textos, novas_notas = [], [], []
    contagem = {nome: 0 for nome, _, _ in AUMENTOS}
    for tema, texto, nota in zip(temas, textos, notas):
        if rng.random() >= fracao:
            continue
        nome, funcao, variacao = rng.choice(AUMENTOS)
        alterado = funcao(texto, rng)
        if not alterado:
            continue
        nova = np.array(nota, dtype=np.float32).copy()
        for comp, delta in variacao.items():
            nova[comp] = max(0.0, nova[comp] + delta)
        novos_temas.append(tema)
        novos_textos.append(alterado)
        novas_notas.append(nova)
        contagem[nome] += 1

    if not novos_textos:
        return temas, textos, notas, contagem
    temas_out = np.concatenate([np.asarray(temas, dtype=object), np.array(novos_temas, dtype=object)])
    textos_out = np.concatenate([np.asarray(textos, dtype=object), np.array(novos_textos, dtype=object)])
    notas_out = np.concatenate([np.asarray(notas, dtype=np.float32), np.stack(novas_notas)])
    return temas_out, textos_out, notas_out, contagem
