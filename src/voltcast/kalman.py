"""Filtro de Kalman (Propuesta B): modelo de nivel y tendencia local, vectorizado sobre celdas.

Estado ``x = [nivel, pendiente]``; ``x_t = F x_{t−1} + w`` con ``F = [[1, 1], [0, 1]]``, ``w ~ N(0, diag(q_nivel, q_pend))``;
observación ``y_t = nivel_t + v``, ``v ~ N(0, r)``. El filtro es **causal**: la estimación en ``t`` usa solo datos ≤ ``t``.
Las varianzas se estiman por máxima verosimilitud sobre train.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

P0_NIVEL, P0_PEND = 1e-2, 1e-6  # varianza inicial (V², (V/día)²) al (re)iniciar un segmento
BURN_IN = 10  # observaciones iniciales de cada segmento excluidas de la verosimilitud


def kalman_filter(y: np.ndarray, blocked: np.ndarray, q_nivel: float, q_pend: float, r: float,
                  return_loglik: bool = False):
    """Filtra ``y [T, C]`` (NaN = sin observación). ``blocked [T, C]`` corta la serie y reinicia el filtro.

    Devuelve el nivel filtrado ``[T, C]`` (NaN en días bloqueados o antes de la primera observación del segmento)
    y, opcionalmente, la log-verosimilitud total.
    """
    T, C = y.shape
    lvl = np.full(C, np.nan)
    slp = np.zeros(C)
    P11, P12, P22 = np.zeros(C), np.zeros(C), np.zeros(C)
    n_obs = np.zeros(C, dtype=int)
    out = np.full((T, C), np.nan)
    ll = 0.0
    for t in range(T):
        blk = blocked[t]
        lvl[blk] = np.nan  # un tramo bloqueado reinicia el segmento
        n_obs[blk] = 0
        activo = ~np.isnan(lvl)
        # Predicción
        lvl_p = lvl + slp
        p11 = P11 + 2 * P12 + P22 + q_nivel
        p12 = P12 + P22
        p22 = P22 + q_pend
        lvl = np.where(activo, lvl_p, lvl)
        P11, P12, P22 = (np.where(activo, a, b) for a, b in ((p11, P11), (p12, P12), (p22, P22)))
        obs = ~np.isnan(y[t]) & ~blk
        # Inicio de segmento: primera observación
        ini = obs & ~activo
        lvl[ini], slp[ini] = y[t, ini], 0.0
        P11[ini], P12[ini], P22[ini] = P0_NIVEL, 0.0, P0_PEND
        # Actualización
        upd = obs & activo
        S = P11 + r
        v = np.where(upd, y[t] - lvl, 0.0)
        K1, K2 = P11 / S, P12 / S
        lvl = np.where(upd, lvl + K1 * v, lvl)
        slp = np.where(upd, slp + K2 * v, slp)
        P11, P12, P22 = (np.where(upd, a, b) for a, b in
                         ((P11 - K1 * P11, P11), (P12 - K1 * P12, P12), (P22 - K2 * P12, P22)))
        n_obs[obs] += 1
        if return_loglik:
            m = upd & (n_obs > BURN_IN)
            ll += float(-0.5 * (np.log(2 * np.pi * S[m]) + v[m] ** 2 / S[m]).sum())
        out[t] = np.where(blk, np.nan, lvl)
    return (out, ll) if return_loglik else out


def fit_mle(y: np.ndarray, blocked: np.ndarray, x0=(-12.0, -18.0, -9.0)) -> dict:
    """Estima (q_nivel, q_pend, r) por máxima verosimilitud conjunta sobre las columnas de ``y`` (log-varianzas)."""
    def nll(theta):
        q1, q2, r = np.exp(np.clip(theta, -30, 0))
        return -kalman_filter(y, blocked, q1, q2, r, return_loglik=True)[1]

    res = minimize(nll, np.array(x0), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2})
    q1, q2, r = np.exp(np.clip(res.x, -30, 0))
    return {"q_nivel": float(q1), "q_pend": float(q2), "r": float(r), "loglik": float(-res.fun),
            "iteraciones": int(res.nit), "convergio": bool(res.success)}
