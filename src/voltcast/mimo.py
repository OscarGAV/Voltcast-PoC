"""Matrices para los modelos del benchmarking que pronostican todas las celdas a la vez (A: MIMO, C: grafo, D: TCN).

Normalización Min-Max por columna (ajustada solo con train) y armado de las matrices diarias de entrada/salida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


class MinMax:
    """Min-Max por columna: ``x' = (x − min) / (max − min)``, ajustado solo con las filas de train válidas."""

    def __init__(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None):
        self.lo, self.hi = lo, hi

    def fit(self, x: np.ndarray, rows: np.ndarray | None = None) -> "MinMax":
        a = x[rows] if rows is not None else x
        self.lo, self.hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
        rango = self.hi - self.lo
        self.hi = np.where(np.isfinite(rango) & (rango > 1e-9), self.hi, self.lo + 1.0)
        self.lo = np.nan_to_num(self.lo)
        self.hi = np.nan_to_num(self.hi, nan=1.0)
        return self

    def transform(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, x):
        return x * (self.hi - self.lo) + self.lo

    def to_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "MinMax":
        return cls(np.asarray(d["lo"]), np.asarray(d["hi"]))


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario (día del año y de la semana como seno/coseno), candidatas para RFR-SHAP."""
    doy = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    dow = 2 * np.pi * dates.dayofweek.to_numpy() / 7
    return pd.DataFrame({"doy_sin": np.sin(doy), "doy_cos": np.cos(doy), "dow_sin": np.sin(dow),
                         "dow_cos": np.cos(dow)}, index=dates)


def calendario_palan(dates: pd.DatetimeIndex) -> np.ndarray:
    """Variables cíclicas de calendario de la Propuesta C (A. Palan & N, 2026): día de la semana, mes y día del año, cada
    uno como seno y coseno ``[T, 6]``. La fuente agrega además el intervalo del día, que no aplica a datos diarios."""
    cols = []
    for valor, periodo in ((dates.dayofweek.to_numpy(), 7), (dates.month.to_numpy() - 1, 12),
                           (dates.dayofyear.to_numpy(), 365.25)):
        ang = 2 * np.pi * valor / periodo
        cols += [np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def causal_ffill(x: np.ndarray) -> np.ndarray:
    """Rellena cada NaN con el último valor anterior (solo pasado, sin fuga). ``x [T, K]``."""
    return pd.DataFrame(x).ffill().to_numpy()


def scaled_inputs(cells: np.ndarray, covs: np.ndarray | None, cell_scaler: MinMax,
                  cov_scaler: MinMax | None = None) -> np.ndarray:
    """Entrada normalizada ``[..., N + K]``: celdas (Min-Max por celda) + covariables (Min-Max). NaN → 0 (mínimo)."""
    x = cell_scaler.transform(cells)
    if covs is not None and covs.shape[-1]:
        x = np.concatenate([x, cov_scaler.transform(covs)], axis=-1)
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def covariate_matrix(selec: list[str], cov_por_el: dict, cal: pd.DataFrame, electrolizadores: list[str]) -> np.ndarray:
    """Covariables de la Propuesta A en el orden usado al entrenar: por electrolizador ``kA_z`` y ``vcel_z`` (las
    elegidas por RFR-SHAP) y luego las variables de calendario elegidas (``doy_sin``, ``doy_cos``, ``dow_sin``, ``dow_cos``).

    ``cov_por_el[el]``: DataFrame con columnas ``kA_z`` y ``vcel_z``; ``cal``: ``calendar_features``.
    """
    cols = [cov_por_el[el][v].to_numpy() for el in electrolizadores for v in ("kA_z", "vcel_z") if v in selec]
    cols += [cal[v].to_numpy() for v in cal.columns if v in selec]
    return np.column_stack(cols) if cols else np.zeros((len(cal), 0))


def modes_matrix(modes: np.ndarray, sel_modes: list[int]) -> np.ndarray:
    """Modos VMD elegidos de cada celda: ``modes [..., N, K]`` → ``[..., N · len(sel_modes)]`` (celda por celda).

    ``sel_modes`` son números de modo desde 1 (1 = el de menor frecuencia).
    """
    m = modes[..., [k - 1 for k in sel_modes]]
    return m.reshape(*m.shape[:-2], -1)
