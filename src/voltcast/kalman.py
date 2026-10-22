"""Filtro de Kalman de la Propuesta B, como Khan et al. (2026, sec. 3.6), vectorizado sobre celdas.

Estado ``x = [nivel, velocidad, aceleración]``; ``x_t = A x_{t−1} + w`` con
``A = [[1, 1, 0.5], [0, 1, 1], [0, 0, 1]]`` y ``w ~ N(0, q·I)``; observación ``y_t = H x_t + v``, ``H = [1, 0, 0]``,
``v ~ N(0, r)``. El filtro es **causal**: la estimación en ``t`` usa solo datos ≤ ``t``.

Adaptaciones al pronóstico de 180 días (docs/hiperparametros.md, 4.3): Khan pronostica un paso y corrige con la
observación de cada paso; aquí, después de ``t0`` no hay observaciones, así que el pronóstico proyecta ``A`` sin
corregir (``nivel + velocidad·h + ½·aceleración·h²``) y se omite la entrada de control (la predicción de la red),
que sin observaciones duplicaría el valor. ``q`` y ``r`` se estiman por máxima verosimilitud con nuestros datos
(los de Khan, Tabla 8, están en kWh²).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

A = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
P0 = np.diag([1e-2, 1e-6, 1e-8])  # varianza inicial al (re)iniciar un segmento (pendiente: inicialización difusa)
BURN_IN = 10  # observaciones iniciales de cada segmento excluidas de la verosimilitud (pendiente de fundamentar)


def kalman_filter(y: np.ndarray, blocked: np.ndarray, q: float, r: float, return_loglik: bool = False):
    """Filtra ``y [T, C]`` (NaN = sin observación). ``blocked [T, C]`` corta la serie y reinicia el filtro.

    Devuelve ``(nivel, velocidad, aceleración)``, cada uno ``[T, C]`` (NaN en días bloqueados o antes de la primera
    observación del segmento); con ``return_loglik`` agrega la log-verosimilitud total.
    """
    T, C = y.shape
    x = np.full((C, 3), np.nan)
    P = np.zeros((C, 3, 3))
    Q = q * np.eye(3)
    n_obs = np.zeros(C, dtype=int)
    out = np.full((3, T, C), np.nan)
    ll = 0.0
    for t in range(T):
        blk = blocked[t]
        x[blk] = np.nan  # un tramo bloqueado reinicia el segmento
        n_obs[blk] = 0
        activo = ~np.isnan(x[:, 0])
        # Predicción: x⁻ = A x, P⁻ = A P Aᵀ + Q
        x[activo] = x[activo] @ A.T
        P[activo] = A @ P[activo] @ A.T + Q
        obs = ~np.isnan(y[t]) & ~blk
        # Inicio de segmento: primera observación
        ini = obs & ~activo
        x[ini] = np.column_stack([y[t, ini], np.zeros(ini.sum()), np.zeros(ini.sum())])
        P[ini] = P0
        # Actualización con H = [1, 0, 0]
        upd = obs & activo
        S = P[:, 0, 0] + r
        v = np.where(upd, y[t] - x[:, 0], 0.0)
        K = P[:, :, 0] / S[:, None]  # [C, 3]
        x[upd] = x[upd] + K[upd] * v[upd, None]
        P[upd] = P[upd] - K[upd][:, :, None] * P[upd][:, 0, None, :]
        n_obs[obs] += 1
        if return_loglik:
            m = upd & (n_obs > BURN_IN)
            ll += float(-0.5 * (np.log(2 * np.pi * S[m]) + v[m] ** 2 / S[m]).sum())
        out[:, t] = np.where(blk[None], np.nan, x.T)
    res = (out[0], out[1], out[2])
    return (*res, ll) if return_loglik else res


def kalman_forecast(level: np.ndarray, vel: np.ndarray, acc: np.ndarray, horizon: int = 180) -> np.ndarray:
    """Pronóstico desde ``t0`` proyectando ``A`` sin corregir: ``nivel + vel·h + ½·acc·h²`` para h = 1..``horizon``."""
    h = np.arange(1, horizon + 1)
    return level[..., None] + vel[..., None] * h + 0.5 * acc[..., None] * h ** 2


def fit_mle(y: np.ndarray, blocked: np.ndarray, x0=(-14.0, -9.0)) -> dict:
    """Estima ``q`` y ``r`` por máxima verosimilitud conjunta sobre las columnas de ``y`` (log-varianzas)."""
    def nll(theta):
        q, r = np.exp(np.clip(theta, -30, 0))
        return -kalman_filter(y, blocked, q, r, return_loglik=True)[3]

    res = minimize(nll, np.array(x0), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2})
    q, r = np.exp(np.clip(res.x, -30, 0))
    return {"q": float(q), "r": float(r), "loglik": float(-res.fun), "iteraciones": int(res.nit),
            "convergio": bool(res.success)}
