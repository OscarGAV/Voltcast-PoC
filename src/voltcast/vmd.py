"""VMD (vmdpy) con hiperparámetros elegidos por optimización bayesiana, como Makri et al. (2026).

- BO-VMD (Makri, sec. 3.2): proceso gaussiano, 30 evaluaciones, K ∈ [2, 15], α ∈ [400, 2000], minimizando el
  error de reconstrucción ‖x − Σ u_k‖ (ec. 14). Se aplica a cada serie que se pronostica (cada celda).
- Uso de los modos: como variables de entrada (Propuesta A, Makri) o, en otros modelos, para reconstruir una
  señal filtrada descartando los ``n_drop`` modos de mayor frecuencia (a decidir por prueba de menor error).
- Las series se descomponen por segmentos en servicio (los tramos ``fuera_servicio`` cortan la serie).
"""
from __future__ import annotations

import numpy as np
import optuna
from vmdpy import VMD

from . import config

# τ = 0 ("para tolerar ruido") y tolerancia ~1e-6: documentación de vmdpy. DC = 0 (sin modo fijo en
# frecuencia 0) e init = 1 (frecuencias centrales iniciales uniformes): pendiente de cita (docs/hiperparametros.md).
TAU, DC, INIT, TOL = 0.0, 0, 1, 1e-6
K_RANGE = (2, 15)  # Makri et al. (2026), sec. 3.2
ALPHA_RANGE = (400.0, 2000.0)
N_DROP = 3  # modos descartados al reconstruir la señal filtrada (prueba de menor error, docs/hiperparametros.md 6.4)


# --------------------------------------------------------------------------- #
# Descomposición y filtrado
# --------------------------------------------------------------------------- #
def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Interpolación lineal de los NaN internos y relleno constante en los extremos (solo para alimentar VMD)."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if ok.all() or not ok.any():
        return x.copy()
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def decompose(x: np.ndarray, K: int, alpha: float) -> np.ndarray:
    """Modos VMD ``[K, len(x)]`` ordenados por frecuencia central creciente.

    vmdpy descarta la última muestra si el largo es impar; aquí se descarta la **primera** para conservar
    el extremo más reciente (clave en el filtrado causal). La primera posición queda NaN en ese caso.
    """
    x = np.asarray(x, dtype=float)
    off = len(x) % 2
    u, _, omega = VMD(x[off:], alpha, TAU, int(K), DC, INIT, TOL)
    u = u[np.argsort(omega[-1])]
    if off:
        u = np.hstack([np.full((u.shape[0], 1), np.nan), u])
    return u


def vmd_filter(x: np.ndarray, K: int, alpha: float, n_drop: int = N_DROP) -> np.ndarray:
    """Señal reconstruida sin los ``n_drop`` modos de mayor frecuencia."""
    u = decompose(fill_gaps(x), K, alpha)
    return u[: K - n_drop].sum(axis=0)


# --------------------------------------------------------------------------- #
# Optimización bayesiana (Makri et al., 2026)
# --------------------------------------------------------------------------- #
def reconstruction_rmse(x: np.ndarray, K: int, alpha: float) -> float:
    """RMSE entre la serie y la suma de sus K modos: el objetivo de la BO-VMD de Makri (ec. 14)."""
    u = decompose(x, K, alpha)
    ok = np.isfinite(u).all(axis=0)
    return float(np.sqrt(np.mean((x[ok] - u[:, ok].sum(axis=0)) ** 2)))


def bo_vmd(x: np.ndarray, n_trials: int = config.BO_TRIALS, seed: int = config.SEED,
           k_range=K_RANGE, alpha_range=ALPHA_RANGE) -> dict:
    """Busca (K, alpha) que minimizan el error de reconstrucción, con optimización bayesiana de proceso gaussiano."""
    xf = fill_gaps(x)

    def objective(trial):
        return reconstruction_rmse(xf, trial.suggest_int("K", *k_range), trial.suggest_float("alpha", *alpha_range))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GPSampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {"K": int(study.best_params["K"]), "alpha": float(study.best_params["alpha"]),
            "rmse_reconstruccion": float(study.best_value)}


# --------------------------------------------------------------------------- #
# Filtrado por segmentos en servicio (train) y causal (validación / inferencia)
# --------------------------------------------------------------------------- #
def _segments(blocked: np.ndarray) -> list[tuple[int, int]]:
    from .cleaning import true_runs
    return true_runs(~np.asarray(blocked, dtype=bool))


def filter_segments(x: np.ndarray, blocked: np.ndarray, K: int, alpha: float,
                    n_drop: int = N_DROP, min_len: int = config.VMD_MIN_LEN) -> np.ndarray:
    """Filtra cada segmento en servicio (no ``blocked``) de al menos ``min_len`` días. El resto queda NaN."""
    out = np.full(len(x), np.nan)
    for a, b in _segments(blocked):
        seg = x[a:b + 1]
        if b - a + 1 >= min_len and np.isfinite(seg).sum() >= min_len // 2:
            out[a:b + 1] = vmd_filter(seg, K, alpha, n_drop)
    return out


def causal_modes(x: np.ndarray, blocked: np.ndarray, t0: int, K: int, alpha: float,
                 history: int = config.VMD_HISTORY, lookback: int = config.LOOKBACK_DAYS,
                 min_len: int = config.VMD_MIN_LEN) -> np.ndarray:
    """Modos VMD ``[K, lookback]`` de los últimos ``lookback`` días, usando solo los ``history`` días hasta ``t0``.

    Solo se usa el segmento en servicio que termina en ``t0``; si ``t0`` está bloqueado o el segmento es
    más corto que ``min_len``, devuelve NaN.
    """
    out = np.full((K, lookback), np.nan)
    a = max(0, t0 - history + 1)
    blk = np.asarray(blocked[a:t0 + 1], dtype=bool)
    if blk[-1]:
        return out
    ini = a + (np.where(blk)[0].max() + 1 if blk.any() else 0)
    seg = x[ini:t0 + 1]
    if len(seg) < min_len or np.isfinite(seg).sum() < min_len // 2:
        return out
    u = decompose(fill_gaps(seg), K, alpha)[:, -lookback:]
    out[:, lookback - u.shape[1]:] = u
    return out


def causal_window(x: np.ndarray, blocked: np.ndarray, t0: int, K: int, alpha: float,
                  n_drop: int = N_DROP, **kw) -> np.ndarray:
    """Señal filtrada causal: suma de los modos de ``causal_modes`` sin los ``n_drop`` de mayor frecuencia."""
    u = causal_modes(x, blocked, t0, K, alpha, **kw)
    return u[: K - n_drop].sum(axis=0) if np.isfinite(u).any() else u[0]
