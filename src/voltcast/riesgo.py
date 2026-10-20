"""Riesgo de intervención de mantenimiento: corrección aditiva al pronóstico de nuestra Bi-LSTM.

El modelo se entrena con la máscara de saltos, así que pronostica la trayectoria **sin** intervenciones. Las
intervenciones se concentran en las celdas de mayor voltaje de su electrolizador, y bajan el voltaje ~0,5 V.
La corrección suma la esperanza histórica de los saltos en ``(t0, t0 + h]`` según el percentil de voltaje de la
celda en ``t0``, atenuada por ``λ`` (elegido por validación cruzada temporal en el notebook 06e):

    V̂(t0 + h) = V̂_red(t0 + h) + λ · E[Σ saltos en (t0, t0 + h] | percentil(t0)]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BINS = np.array([0, .5, .8, .9, .95, 1.0001])  # percentil de voltaje dentro del electrolizador


def percentil(v_valido: np.ndarray, cells: list[str], dias: int = 7) -> np.ndarray:
    """``[T, C]``: percentil del voltaje real medio de los últimos ``dias`` (causal) dentro de su electrolizador."""
    m = pd.DataFrame(v_valido).rolling(dias, min_periods=1).mean()
    out = np.full(v_valido.shape, np.nan)
    for el in sorted({c[0] for c in cells}):
        j = [i for i, c in enumerate(cells) if c[0] == el]
        out[:, j] = m.iloc[:, j].rank(axis=1, pct=True).to_numpy()
    return out


def saltos_acumulados(dates: pd.DatetimeIndex, cells: list[str], saltos: pd.DataFrame) -> np.ndarray:
    """``[T, C]``: suma acumulada de ``delta_V`` de los saltos de nivel hasta cada fecha (inclusive)."""
    J = np.zeros((len(dates), len(cells)))
    pos, col = {d: i for i, d in enumerate(dates)}, {c: j for j, c in enumerate(cells)}
    for f, c, dv in saltos[["fecha", "celda", "delta_V"]].itertuples(index=False):
        if pd.Timestamp(f) in pos and c in col:
            J[pos[pd.Timestamp(f)], col[c]] += dv
    return np.cumsum(J, axis=0)


def ajustar(pct: np.ndarray, acum: np.ndarray, hasta: int, horizon: int, t_min: int = 30) -> np.ndarray:
    """``[H, n_bins]``: media de Σ saltos en ``(t, t + h]`` por bin de percentil, con orígenes ``t ∈ [t_min, hasta − h]``
    (todos los objetivos ≤ ``hasta``: solo usa historia disponible)."""
    b = np.digitize(pct, BINS) - 1
    c = np.zeros((horizon, len(BINS) - 1))
    for h in range(1, horizon + 1):
        t = np.arange(t_min, hasta - h + 1)
        if len(t) == 0:
            continue
        d, bb = acum[t + h] - acum[t], b[t]
        for k in range(len(BINS) - 1):
            sel = bb == k
            c[h - 1, k] = d[sel].mean() if sel.any() else 0.0
    return c


def aplicar(pred: np.ndarray, pct0: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    """``pred [O, C, H] + λ · c[h, bin(pct0)]`` con ``pct0 [O, C]`` (sin percentil → bin central)."""
    b = np.clip(np.digitize(np.nan_to_num(pct0, nan=0.5), BINS) - 1, 0, len(BINS) - 2)
    return pred + lam * c[:, b].transpose(1, 2, 0)
