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
def persistence(last: np.ndarray, horizon: int = config.HORIZON_DAYS) -> np.ndarray:
    """Persistencia: repite un valor (``last [...]``) en todo el horizonte → ``[..., H]``."""
    return np.repeat(last[..., None], horizon, axis=-1)


def last_observed(window: np.ndarray) -> np.ndarray:
    """Último valor no NaN de cada ventana ``[..., L]`` (el voltaje real de t0, o el más reciente disponible)."""
    ok = np.isfinite(window)
    idx = np.where(ok.any(-1), window.shape[-1] - 1 - np.argmax(ok[..., ::-1], axis=-1), 0)
    out = np.take_along_axis(window, idx[..., None], axis=-1)[..., 0]
    return np.where(ok.any(-1), out, np.nan)


def linear_trend(window: np.ndarray, horizon: int = config.HORIZON_DAYS) -> np.ndarray:
    """Recta ``y = a + b·t`` por mínimos cuadrados sobre los valores no NaN de la ventana ``[..., L]``,
    extrapolada ``H`` días (t0 = 0). Con menos de 2 puntos devuelve NaN."""
    L = window.shape[-1]
    t = np.broadcast_to(np.arange(L) - (L - 1), window.shape).astype(float)
    ok = np.isfinite(window)
    n = ok.sum(-1)
    tm = np.where(ok, t, 0).sum(-1) / np.maximum(n, 1)
    ym = np.where(ok, window, 0).sum(-1) / np.maximum(n, 1)
    dt = np.where(ok, t - tm[..., None], 0)
    dy = np.where(ok, window - ym[..., None], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (dt * dy).sum(-1) / (dt ** 2).sum(-1)
    a = ym - b * tm
    a = np.where(n >= 2, a, np.nan)
    h = np.arange(1, horizon + 1)
    return a[..., None] + b[..., None] * h


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _masked(x, mask):
    return np.where(mask, x, np.nan)


def rmse(pred, real, mask, axis=None):
    return np.sqrt(np.nanmean(_masked((pred - real) ** 2, mask), axis=axis))


def mse(pred, real, mask, axis=None):
    return np.nanmean(_masked((pred - real) ** 2, mask), axis=axis)


def r2_global(pred, real, mask) -> float:
    """R² sobre todos los objetivos evaluables juntos: 1 − Σ(y − ŷ)² / Σ(y − ȳ)²."""
    p, r = pred[mask], real[mask]
    return float(1 - ((r - p) ** 2).sum() / ((r - r.mean()) ** 2).sum())


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


def summary_at(preds: dict, real, mask, h: int | None) -> pd.DataFrame:
    """Métricas para cada modelo a horizonte ``h`` (1-indexado) o, con ``h=None``, sobre los horizontes 1–H juntos.

    MAE, RMSE en V; MSE en V²; MAPE en %; R² global (todos los objetivos) y R² por celda (media y mediana).
    """
    filas = []
    for nombre, p in preds.items():
        if h is None:
            pk, rk, mk = p, real, mask
            r2c = np.nanmean(np.stack([r2_per_cell(p[..., k], real[..., k], mask[..., k])
                                       for k in range(p.shape[-1])]), axis=0)
        else:
            k = h - 1
            pk, rk, mk = p[..., k], real[..., k], mask[..., k]
            r2c = r2_per_cell(pk, rk, mk)
        filas.append({
            "modelo": nombre, "horizonte": "1–180" if h is None else h, "n": int(mk.sum()),
            "MAE_V": float(mae(pk, rk, mk)), "MSE_V2": float(mse(pk, rk, mk)), "RMSE_V": float(rmse(pk, rk, mk)),
            "MAPE_%": float(mape(pk, rk, mk)), "R2_global": r2_global(pk, rk, mk),
            "R2_medio": float(np.nanmean(r2c)), "R2_mediana": float(np.nanmedian(r2c)),
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
