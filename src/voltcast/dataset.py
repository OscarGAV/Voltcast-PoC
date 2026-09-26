"""Ancla, objetivo normalizado y orígenes de pronóstico.

(El Dataset/DataLoader de PyTorch se agrega en el notebook 06.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def pivot(df: pd.DataFrame, value: str, cells: list[str] | None = None) -> pd.DataFrame:
    """Formato largo → tabla ``fecha × celda``."""
    w = df.pivot(index="fecha", columns="celda", values=value)
    w.columns = w.columns.astype(str)
    return w[cells] if cells is not None else w


def anchor(vf: pd.DataFrame | np.ndarray, days: int = config.ANCHOR_DAYS):
    """Ancla en cada fecha: media de los últimos ``days`` días de la señal filtrada (exige los ``days`` valores)."""
    if isinstance(vf, np.ndarray):
        return pd.DataFrame(vf).rolling(days, min_periods=days).mean().to_numpy()
    return vf.rolling(days, min_periods=days).mean()


def anchor_from_window(window: np.ndarray, days: int = config.ANCHOR_DAYS) -> np.ndarray:
    """Ancla a partir de ventanas ``[..., L]`` (último eje = tiempo): media de los últimos ``days`` valores."""
    w = window[..., -days:]
    a = w.mean(axis=-1)
    return np.where(np.isfinite(w).all(axis=-1), a, np.nan)


def origin_ok(anc: pd.DataFrame, oos: pd.DataFrame, lookback: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Un origen ``t0`` es usable si tiene ancla y la ventana de entrada no toca un tramo fuera de servicio."""
    toca_oos = oos.astype(float).rolling(lookback, min_periods=1).max().astype(bool)
    return anc.notna() & ~toca_oos


def delta_stats(v_real: pd.DataFrame, valido: pd.DataFrame, anc: pd.DataFrame, ok_t0: pd.DataFrame,
                horizon: int = config.HORIZON_DAYS) -> tuple[float, pd.DataFrame]:
    """Estadísticos de Δ(t0, h) = V_real(t0 + h) − ancla(t0) para h = 1..``horizon``.

    Solo cuenta objetivos con calidad válida y orígenes usables; todo dentro del rango de fechas recibido
    (pasar solo train). Devuelve σ_Δ global y una tabla por horizonte (n, media, desvío).
    """
    V = np.where(valido.to_numpy(), v_real.to_numpy(), np.nan)
    A = np.where(ok_t0.to_numpy(), anc.to_numpy(), np.nan)
    filas, n_tot, s_tot, ss_tot = [], 0, 0.0, 0.0
    for h in range(1, horizon + 1):
        d = V[h:] - A[:-h]
        d = d[np.isfinite(d)]
        n, s, ss = d.size, float(d.sum()), float((d ** 2).sum())
        n_tot, s_tot, ss_tot = n_tot + n, s_tot + s, ss_tot + ss
        filas.append({"h": h, "n": n, "media": s / n if n else np.nan,
                      "desvio": np.sqrt(max(ss / n - (s / n) ** 2, 0)) if n else np.nan})
    sigma = float(np.sqrt(ss_tot / n_tot - (s_tot / n_tot) ** 2))
    return sigma, pd.DataFrame(filas).set_index("h")


def validation_origins(split: dict, stride: int = config.EVAL_STRIDE) -> pd.DatetimeIndex:
    """Orígenes rolling-origin: desde el último día de train hasta el penúltimo de validación, cada ``stride`` días.

    ``t0`` es el último día observado; el primer objetivo es ``t0 + 1``.
    """
    ini = pd.Timestamp(split["train_hasta"])
    fin = pd.Timestamp(split["val_hasta"]) - pd.Timedelta(days=1)
    return pd.date_range(ini, fin, freq=f"{stride}D", name="fecha_origen")
