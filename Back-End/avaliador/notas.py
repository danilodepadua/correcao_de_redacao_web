"""Conversão da previsão contínua em notas 0, 40, ..., 200, calibração dos cortes e QWK."""
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Cortes padrão: múltiplo de 40 mais próximo
DEFAULT_CUTS = [20.0, 60.0, 100.0, 140.0, 180.0]
NOTAS = np.array([0, 40, 80, 120, 160, 200])


def apply_cuts(preds_0_200: np.ndarray, cuts) -> np.ndarray:
    """Converte previsões contínuas (escala 0–200) em notas, dados 5 cortes crescentes."""
    return NOTAS[np.searchsorted(np.asarray(cuts), preds_0_200, side="right")]


def post_process_scores(preds_array):
    """Arredondamento simples de previsões na escala 0–1 para múltiplos de 40 (0–200)."""
    raw_scores = preds_array * 200.0
    clipped = np.clip(raw_scores, 0, 200)
    rounded = (np.round(clipped / 40.0) * 40).astype(int)
    return rounded


def qwk(y_true, y_pred):
    """Quadratic Weighted Kappa: concordância com as notas humanas (1 = perfeita)."""
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
