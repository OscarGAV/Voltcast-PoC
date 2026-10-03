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
    """Covariables de la Propuesta A en el orden usado al entrenar: por electrolizador [kA_z, vcel_z] (las
    seleccionadas por RFR-SHAP) y luego el calendario [doy_sin, doy_cos, dow_sin, dow_cos] seleccionado.

    ``cov_por_el[el]``: DataFrame con columnas ``kA_z`` y ``vcel_z``; ``cal``: ``calendar_features``.
    """
    cols = []
    for el in electrolizadores:
        if "kA" in selec:
            cols.append(cov_por_el[el]["kA_z"].to_numpy())
        if "vcel" in selec:
            cols.append(cov_por_el[el]["vcel_z"].to_numpy())
    for g, nombres in (("doy", ["doy_sin", "doy_cos"]), ("dow", ["dow_sin", "dow_cos"])):
        if g in selec:
            cols += [cal[n].to_numpy() for n in nombres]
    return np.column_stack(cols) if cols else np.zeros((len(cal), 0))
