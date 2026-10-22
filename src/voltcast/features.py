"""RFR-SHAP (Propuesta A): selección explicable de covariables con Random Forest + valores SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config


def rfr_shap(X: pd.DataFrame, y: pd.Series, n_explain: int = 1000, seed: int = config.SEED,
             n_jobs: int = config.N_JOBS) -> tuple[pd.DataFrame, np.ndarray]:
    """Random Forest ``y ~ X`` (Makri et al., 2026: 100 árboles; el resto, valores por defecto de scikit-learn) e
    importancia SHAP media |φ| de cada variable. Devuelve (tabla ordenada, |φ| ``[n_explain, n_vars]``).

    SHAP se calcula en paralelo por bloques de muestras (el resultado es el mismo que en un solo proceso).
    """
    import shap
    from joblib import Parallel, delayed

    rf = RandomForestRegressor(n_estimators=config.RFR_TREES, n_jobs=n_jobs, random_state=seed).fit(X, y)
    muestra = X.sample(min(n_explain, len(X)), random_state=seed).to_numpy()
    from joblib import effective_n_jobs

    bloques = np.array_split(muestra, effective_n_jobs(n_jobs))
    phi = np.vstack(Parallel(n_jobs=n_jobs)(delayed(lambda b: shap.TreeExplainer(rf).shap_values(b))(b) for b in bloques))
    imp = pd.DataFrame({"variable": X.columns, "shap_medio_abs": np.abs(phi).mean(axis=0)})
    imp["porcentaje"] = 100 * imp["shap_medio_abs"] / imp["shap_medio_abs"].sum()
    imp["r2_train"] = rf.score(X, y)
    return imp.sort_values("shap_medio_abs", ascending=False, ignore_index=True), np.abs(phi)


def top_k(abs_phi: np.ndarray, columns, k: int = config.RFR_TOP) -> list[str]:
    """Las ``k`` variables de mayor |φ| medio."""
    return [columns[i] for i in np.argsort(-abs_phi.mean(axis=0))[:k]]
