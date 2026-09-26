"""VMD (vmdpy) con hiperparámetros elegidos por optimización bayesiana (optuna, TPE).

- Objetivo de la BO: mínima entropía de envolvente de los modos (Hilbert).
- Filtrado: se reconstruye la señal descartando los ``n_drop`` modos de mayor frecuencia central.
- Las series se filtran por segmentos en servicio (los tramos ``fuera_servicio`` cortan la serie).
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from scipy.signal import hilbert
from vmdpy import VMD

from . import config

TAU, DC, INIT, TOL = 0.0, 0, 1, 1e-7
K_RANGE = (3, 10)
ALPHA_RANGE = (500.0, 5000.0)
N_DROP = 1  # modos de alta frecuencia descartados


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


def envelope_entropy(u: np.ndarray) -> np.ndarray:
    """Entropía de envolvente de cada modo: E = −Σ p·ln p, con p = |hilbert(u)| normalizada."""
    out = []
    for m in u:
        m = m[~np.isnan(m)]
        a = np.abs(hilbert(m))
        p = a / a.sum()
        out.append(float(-(p * np.log(p + 1e-300)).sum()))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Optimización bayesiana
# --------------------------------------------------------------------------- #
def bo_vmd(x: np.ndarray, n_trials: int = config.BO_TRIALS, seed: int = config.SEED,
           k_range=K_RANGE, alpha_range=ALPHA_RANGE) -> dict:
    """Busca (K, alpha) que minimizan la mínima entropía de envolvente de los modos."""
    xf = fill_gaps(x)

    def objective(trial):
        K = trial.suggest_int("K", *k_range)
        alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        return float(envelope_entropy(decompose(xf, K, alpha)).min())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params"))
    return {"K": int(study.best_params["K"]), "alpha": float(study.best_params["alpha"]),
            "entropia": float(study.best_value), "trials": trials}


def select_bo_sample(niveles: pd.DataFrame, n: int = config.BO_SAMPLE_CELLS, seed: int = config.SEED) -> list[str]:
    """Muestra estratificada por electrolizador, posición en el stack (inicio/medio/final) y nivel de voltaje (terciles).

    ``niveles``: columnas ``celda, electrolizador, posicion, nivel`` solo con celdas elegibles
    (sin tramos ``fuera_servicio``).
    """
    rng = np.random.default_rng(seed)
    elegidas = []
    por_el = n // niveles["electrolizador"].nunique()
    for el, g in niveles.groupby("electrolizador", observed=True):
        g = g.assign(tramo=pd.cut(g["posicion"], 3, labels=["inicio", "medio", "final"]),
                     tercil=pd.qcut(g["nivel"], 3, labels=["bajo", "medio", "alto"]))
        sel = [grp["celda"].iloc[rng.integers(len(grp))]
               for _, grp in g.groupby(["tramo", "tercil"], observed=True) if len(grp)]
        resto = g.loc[~g["celda"].isin(sel), "celda"].to_numpy()
        sel += list(rng.choice(resto, size=max(0, por_el - len(sel)), replace=False))
        elegidas += sel[:por_el]
    return elegidas


def choose_params(res: pd.DataFrame, max_k_diff: int = 1, max_alpha_ratio: float = 2.0) -> dict:
    """Par común (mediana) si los electrolizadores son parecidos; si no, un par por electrolizador.

    Criterio de homogeneidad: |Δ mediana K| ≤ ``max_k_diff`` y cociente de medianas de alpha ≤ ``max_alpha_ratio``.
    """
    med = res.groupby("electrolizador")[["K", "alpha"]].median()
    dk = float(med["K"].max() - med["K"].min())
    ratio = float(med["alpha"].max() / med["alpha"].min())
    homogeneo = dk <= max_k_diff and ratio <= max_alpha_ratio
    if homogeneo:
        comun = {"K": int(round(res["K"].median())), "alpha": float(res["alpha"].median())}
        por_el = {el: comun for el in med.index}
    else:
        por_el = {el: {"K": int(round(r["K"])), "alpha": float(r["alpha"])} for el, r in med.iterrows()}
    return {"modo": "comun" if homogeneo else "por_electrolizador", "params": por_el,
            "diferencia_K": dk, "cociente_alpha": ratio, "n_drop": N_DROP}


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


def causal_window(x: np.ndarray, blocked: np.ndarray, t0: int, K: int, alpha: float,
                  n_drop: int = N_DROP, history: int = config.VMD_HISTORY,
                  lookback: int = config.LOOKBACK_DAYS, min_len: int = config.VMD_MIN_LEN) -> np.ndarray:
    """Últimos ``lookback`` días filtrados con VMD usando solo los ``history`` días hasta ``t0`` (inclusive).

    Solo se usa el segmento en servicio que termina en ``t0``; si ``t0`` está bloqueado o el segmento es
    más corto que ``min_len``, devuelve NaN.
    """
    out = np.full(lookback, np.nan)
    a = max(0, t0 - history + 1)
    blk = np.asarray(blocked[a:t0 + 1], dtype=bool)
    if blk[-1]:
        return out
    ini = a + (np.where(blk)[0].max() + 1 if blk.any() else 0)
    seg = x[ini:t0 + 1]
    if len(seg) < min_len or np.isfinite(seg).sum() < min_len // 2:
        return out
    f = vmd_filter(seg, K, alpha, n_drop)[-lookback:]
    out[lookback - len(f):] = f
    return out
