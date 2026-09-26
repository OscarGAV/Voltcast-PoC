"""Reglas de la capa Silver (R1–R9 del notebook 03)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .io import DATE_COL, KA_COL, VOLTS_COL

# Códigos de calidad, de menor a mayor precedencia al asignarlos
CALIDAD = ["ok", "interpolado", "hueco", "atipico", "paro", "fuera_servicio"]
CALIDAD_VALIDA = ["ok", "interpolado"]  # fechas usadas en métricas


def cell_name(el: str, n: int | str) -> str:
    return f"{el}_c{int(n):03d}"


# --------------------------------------------------------------------------- #
# R1–R2: nombres, fecha calendario, duplicados y frecuencia diaria
# --------------------------------------------------------------------------- #
def daily_wide(bronze: pd.DataFrame, el: str, cell_cols: list[str], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Bronze de un electrolizador → tabla diaria continua con columnas ``{el}_V_total``, ``{el}_kA``, ``{el}_c001``…

    Descarta ``Average`` y las columnas de linaje. En fechas duplicadas conserva el último registro del día.
    """
    df = bronze.sort_values(DATE_COL, kind="stable")
    df = df.assign(fecha=df[DATE_COL].dt.normalize()).drop_duplicates("fecha", keep="last").set_index("fecha")
    rename = {VOLTS_COL: f"{el}_V_total", KA_COL: f"{el}_kA", **{c: cell_name(el, c) for c in cell_cols}}
    out = df[list(rename)].rename(columns=rename).reindex(calendar)
    out.index.name = "fecha"
    return out


def calendar_from(*bronzes: pd.DataFrame) -> pd.DatetimeIndex:
    fechas = pd.concat([b[DATE_COL].dt.normalize() for b in bronzes])
    return pd.date_range(fechas.min(), fechas.max(), freq="D", name="fecha")


# --------------------------------------------------------------------------- #
# Utilidades de tramos
# --------------------------------------------------------------------------- #
def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Tramos consecutivos de True como pares (inicio, fin) inclusivos."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def interpolate_short_gaps(s: pd.Series, max_days: int, blocked: np.ndarray | None = None) -> tuple[pd.Series, np.ndarray]:
    """Interpola linealmente los huecos interiores de hasta ``max_days`` días.

    Los huecos más largos, los de los extremos y los que tocan un día ``blocked`` quedan como NaN.
    Devuelve la serie y la máscara de valores interpolados.
    """
    vals = s.to_numpy(dtype=float, copy=True)
    filled = np.zeros(len(vals), dtype=bool)
    for a, b in true_runs(np.isnan(vals)):
        if a == 0 or b == len(vals) - 1 or b - a + 1 > max_days:
            continue
        if blocked is not None and blocked[a:b + 1].any():
            continue
        vals[a:b + 1] = np.interp(np.arange(a, b + 1), [a - 1, b + 1], [vals[a - 1], vals[b + 1]])
        filled[a:b + 1] = True
    return pd.Series(vals, index=s.index, name=s.name), filled


# --------------------------------------------------------------------------- #
# R5: celdas fuera de servicio
# --------------------------------------------------------------------------- #
def out_of_service_mask(v: pd.DataFrame, ka: pd.Series,
                        min_days: int = config.OOS_MIN_DAYS,
                        merge_days: int = config.OOS_MERGE_DAYS) -> pd.DataFrame:
    """Tramos fuera de servicio por celda.

    Un día "informativo" tiene kA normal y dato en la mayoría de las celdas. En esos días la celda está
    "mal" si V < ``V_OUT_OF_SERVICE`` (incluye negativos) o si le falta el dato. Los tramos de días malos
    se unen si los separan hasta ``merge_days`` días informativos buenos, y se conservan si abarcan al
    menos ``min_days`` días calendario. Los días de paro o sin registro dentro del tramo no lo cortan.
    """
    informativo = (ka >= config.KA_SHUTDOWN) & (v.notna().sum(axis=1) > v.shape[1] // 2)
    pos = np.where(informativo.to_numpy())[0]
    malo = ((v < config.V_OUT_OF_SERVICE) | v.isna()).to_numpy()[pos]
    out = np.zeros(v.shape, dtype=bool)
    for j in range(v.shape[1]):
        runs = true_runs(malo[:, j])
        merged: list[list[int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] - 1 <= merge_days:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            ini, fin = pos[a], pos[b]
            if (v.index[fin] - v.index[ini]).days + 1 >= min_days:
                out[ini:fin + 1, j] = True
    return pd.DataFrame(out, index=v.index, columns=v.columns)


# --------------------------------------------------------------------------- #
# R3–R6: máscara de calidad e interpolación
# --------------------------------------------------------------------------- #
def quality(v: pd.DataFrame, ka: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (voltaje limpio, código de calidad) para las celdas de un electrolizador.

    - ``fuera_servicio``: tramos de ``out_of_service_mask`` → NaN.
    - ``paro``: kA < ``KA_SHUTDOWN`` → NaN (interpolado si el paro dura ≤ ``MAX_INTERP_DAYS``, pero conserva el código).
    - ``atipico``: V > ``V_MAX_PLAUSIBLE`` o V < ``V_OUT_OF_SERVICE`` aislados → NaN.
    - ``hueco``: sin dato. Los huecos y atípicos de ≤ ``MAX_INTERP_DAYS`` días pasan a ``interpolado``.
    """
    oos = out_of_service_mask(v, ka)
    paro = np.broadcast_to((ka < config.KA_SHUTDOWN).to_numpy()[:, None], v.shape)
    raw = v.to_numpy()
    atip = ((raw > config.V_MAX_PLAUSIBLE) | (raw < config.V_OUT_OF_SERVICE)) & ~paro
    code = np.zeros(v.shape, dtype=np.int8)  # ok
    code[np.isnan(raw)] = CALIDAD.index("hueco")
    code[atip] = CALIDAD.index("atipico")
    code[paro] = CALIDAD.index("paro")
    code[oos.to_numpy()] = CALIDAD.index("fuera_servicio")

    clean = v.where(code == 0)
    blocked = code == CALIDAD.index("fuera_servicio")
    for j, c in enumerate(v.columns):
        clean[c], filled = interpolate_short_gaps(clean[c], config.MAX_INTERP_DAYS, blocked[:, j])
        reclasificar = filled & np.isin(code[:, j], [CALIDAD.index("hueco"), CALIDAD.index("atipico")])
        code[reclasificar, j] = CALIDAD.index("interpolado")
    return clean, pd.DataFrame(code, index=v.index, columns=v.columns)


# --------------------------------------------------------------------------- #
# R9: saltos de nivel
# --------------------------------------------------------------------------- #
def level_shifts(clean: pd.DataFrame, code: pd.DataFrame,
                 threshold: float = config.STEP_THRESHOLD_V, window: int = config.STEP_WINDOW_DAYS) -> pd.DataFrame:
    """Saltos de nivel por celda: |mediana(7 días siguientes) − mediana(7 días previos)| > ``threshold``.

    Solo usa fechas ``ok``/``interpolado``. Cada racha de días que superan el umbral es un evento;
    ``fecha`` es el primer día del nuevo nivel y ``delta_V`` el cambio máximo de la racha.
    Uso exclusivo para diagnóstico: **no** es una entrada del modelo.
    """
    valid = clean.where(code.isin([CALIDAD.index(c) for c in CALIDAD_VALIDA]))
    min_p = max(1, window // 2 + 1)
    antes = valid.rolling(window, min_periods=min_p).median()
    despues = valid[::-1].rolling(window, min_periods=min_p).median()[::-1].shift(-1)
    delta = (despues - antes).to_numpy()
    eventos = []
    for j, c in enumerate(clean.columns):
        d = delta[:, j]
        for a, b in true_runs(np.abs(np.nan_to_num(d)) > threshold):
            k = a + int(np.nanargmax(np.abs(d[a:b + 1])))
            if k + 1 < len(clean):
                eventos.append({"fecha": clean.index[k + 1], "celda": c, "delta_V": float(d[k])})
    return pd.DataFrame(eventos, columns=["fecha", "celda", "delta_V"])


# --------------------------------------------------------------------------- #
# Ensamblado de un electrolizador → formato largo
# --------------------------------------------------------------------------- #
def silver_electrolizador(wide: pd.DataFrame, el: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica R3–R7 y R9 a la tabla diaria de un electrolizador.

    Devuelve (formato largo, saltos de nivel).
    """
    cells = [c for c in wide.columns if c.startswith(f"{el}_c")]
    ka_raw = wide[f"{el}_kA"]
    clean, code = quality(wide[cells], ka_raw)
    ka, _ = interpolate_short_gaps(ka_raw, config.MAX_INTERP_DAYS)
    vtot, _ = interpolate_short_gaps(wide[f"{el}_V_total"], config.MAX_INTERP_DAYS)
    n_srv = (code != CALIDAD.index("fuera_servicio")).sum(axis=1)

    largo = pd.DataFrame({
        "fecha": np.repeat(wide.index.to_numpy(), len(cells)),
        "electrolizador": el,
        "celda": np.tile(cells, len(wide)),
        "voltaje": clean.to_numpy().ravel(),
        "kA": np.repeat(ka.to_numpy(), len(cells)),
        "V_total": np.repeat(vtot.to_numpy(), len(cells)),
        "n_en_servicio": np.repeat(n_srv.to_numpy(), len(cells)).astype(np.int16),
        "calidad": pd.Categorical.from_codes(code.to_numpy().ravel(), categories=CALIDAD),
    })
    saltos = level_shifts(clean, code).assign(electrolizador=el)
    return largo, saltos[["fecha", "electrolizador", "celda", "delta_V"]]


def finalize_long(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df["electrolizador"] = df["electrolizador"].astype("category")
    df["celda"] = df["celda"].astype("category")
    df["calidad"] = pd.Categorical(df["calidad"], categories=CALIDAD)
    return df.sort_values(["electrolizador", "celda", "fecha"], ignore_index=True)


# --------------------------------------------------------------------------- #
# R8: split temporal
# --------------------------------------------------------------------------- #
def temporal_split(calendar: pd.DatetimeIndex, val_fraction: float = config.VAL_FRACTION) -> dict:
    n = len(calendar)
    n_val = round(n * val_fraction)
    train, val = calendar[: n - n_val], calendar[n - n_val:]
    return {
        "val_fraction": val_fraction,
        "n_dias": n,
        "n_train": len(train),
        "n_val": n_val,
        "train_desde": str(train[0].date()),
        "train_hasta": str(train[-1].date()),
        "val_desde": str(val[0].date()),
        "val_hasta": str(val[-1].date()),
        "fecha_corte": str(val[0].date()),  # primera fecha de validación
    }
