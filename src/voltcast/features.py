"""RFR-SHAP (Propuesta A): selección explicable de covariables con Random Forest + valores SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config


def rfr_shap(X: pd.DataFrame, y: pd.Series, n_explain: int = 2000, seed: int = config.SEED) -> pd.DataFrame:
    """Ajusta un Random Forest ``y ~ X`` y devuelve la importancia SHAP media |φ| de cada variable (y su %)."""
    import shap

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    rf.fit(X, y)
    muestra = X.sample(min(n_explain, len(X)), random_state=seed)
    phi = shap.TreeExplainer(rf).shap_values(muestra)
    imp = pd.DataFrame({"variable": X.columns, "shap_medio_abs": np.abs(phi).mean(axis=0)})
    imp["porcentaje"] = 100 * imp["shap_medio_abs"] / imp["shap_medio_abs"].sum()
    imp["r2_oob_aprox"] = rf.score(X, y)
    return imp.sort_values("shap_medio_abs", ascending=False, ignore_index=True)


def select_covariates(imp: pd.DataFrame, grupos: dict[str, list[str]], umbral_pct: float = 5.0) -> list[str]:
    """Covariables cuyo grupo de variables (p. ej. ``doy`` = [doy_sin, doy_cos]) suma ≥ ``umbral_pct`` % de SHAP."""
    pct = imp.set_index("variable")["porcentaje"]
    return [g for g, cols in grupos.items() if pct.reindex(cols).fillna(0).sum() >= umbral_pct]
