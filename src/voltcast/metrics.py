"""Baselines y métricas de evaluación (en voltios, contra la señal real sin filtrar).

Convención de arreglos: ``[O, C, H]`` = orígenes × celdas × horizontes; ``mask`` marca los objetivos evaluables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def persistence(anchor: np.ndarray, horizon: int = config.HORIZON_DAYS) -> np.ndarray:
    """Persistencia: el nivel del ancla se mantiene en todo el horizonte. ``[...] → [..., H]``."""
    return np.repeat(anchor[..., None], horizon, axis=-1)


def linear_trend(window: np.ndarray, horizon: int = config.HORIZON_DAYS) -> np.ndarray:
    """Tendencia lineal por mínimos cuadrados sobre la ventana de entrada ``[..., L]``, extrapolada ``H`` días."""
    L = window.shape[-1]
    t = np.arange(L) - (L - 1)  # t0 = 0
    tc = t - t.mean()
    y = window
    b = (tc * (y - y.mean(axis=-1, keepdims=True))).sum(-1) / (tc ** 2).sum()
    a = y.mean(-1) - b * t.mean()  # valor de la recta en t0
    h = np.arange(1, horizon + 1)
    return a[..., None] + b[..., None] * h


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _masked(x, mask):
    return np.where(mask, x, np.nan)


def rmse(pred, real, mask, axis=None):
    return np.sqrt(np.nanmean(_masked((pred - real) ** 2, mask), axis=axis))


def mae(pred, real, mask, axis=None):
    return np.nanmean(_masked(np.abs(pred - real), mask), axis=axis)


def mape(pred, real, mask, axis=None):
    return 100 * np.nanmean(_masked(np.abs(pred - real) / np.abs(real), mask), axis=axis)


def r2_per_cell(pred, real, mask):
    """R² de cada celda sobre sus orígenes, para un horizonte fijo: entradas ``[O, C]`` → ``[C]``.

    Con pocos orígenes y un nivel casi constante, el denominador es pequeño y el R² puede ser muy negativo.
    """
    p, r = _masked(pred, mask), _masked(real, mask)
    sse = np.nansum((p - r) ** 2, axis=0)
    sst = np.nansum((r - np.nanmean(r, axis=0)) ** 2, axis=0)
    n = mask.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((n >= 3) & (sst > 0), 1 - sse / sst, np.nan)


def summary_at(preds: dict, real, mask, h: int) -> pd.DataFrame:
    """Tabla de métricas a horizonte ``h`` (1-indexado) para cada modelo, en V (RMSE/MAE) y % (MAPE)."""
    k = h - 1
    filas = []
    for nombre, p in preds.items():
        pk, rk, mk = p[..., k], real[..., k], mask[..., k]
        r2 = r2_per_cell(pk, rk, mk)
        filas.append({
            "modelo": nombre, "horizonte": h, "n": int(mk.sum()),
            "RMSE_V": float(rmse(pk, rk, mk)), "MAE_V": float(mae(pk, rk, mk)), "MAPE_%": float(mape(pk, rk, mk)),
            "R2_medio": float(np.nanmean(r2)), "R2_mediana": float(np.nanmedian(r2)),
        })
    return pd.DataFrame(filas)


def bootstrap_rmse_diff(pred_a, pred_b, real, mask, h: int, n_boot: int = 1000, seed: int = config.SEED):
    """IC 95 % de RMSE(a) − RMSE(b) a horizonte ``h``, remuestreando **orígenes** (bloques)."""
    rng = np.random.default_rng(seed)
    k = h - 1
    ea = _masked((pred_a[..., k] - real[..., k]) ** 2, mask[..., k])
    eb = _masked((pred_b[..., k] - real[..., k]) ** 2, mask[..., k])
    sa, sb, n = np.nansum(ea, 1), np.nansum(eb, 1), mask[..., k].sum(1)
    ok = n > 0
    sa, sb, n = sa[ok], sb[ok], n[ok]
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    d = np.sqrt(sa[idx].sum(1) / n[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / n[idx].sum(1))
    return float(np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())), np.percentile(d, [2.5, 97.5])
