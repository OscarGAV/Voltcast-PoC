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


# --------------------------------------------------------------------------- #
# Tensores para el modelo (todo cabe en memoria de la GPU: ~2000 días × 362 celdas)
# --------------------------------------------------------------------------- #
FEATURES = ["v_rel", "kA_z", "vcel_z", "mascara"]


def build_inputs(vf_win, valid_win, ka_win, vcel_win, sigma: float, anchor_days: int = config.ANCHOR_DAYS):
    """Ventanas ``[N, L]`` → (entrada ``[N, L, 4]``, ancla ``[N]``). Funciona con tensores de torch.

    Canales: voltaje filtrado relativo al ancla ÷ σ_Δ, kA_z, vcel_z y máscara de calidad válida.
    """
    import torch

    anc = vf_win[:, -anchor_days:].mean(dim=1)
    v_rel = (vf_win - anc[:, None]) / sigma
    x = torch.stack([v_rel, torch.nan_to_num(ka_win), torch.nan_to_num(vcel_win), valid_win.float()], dim=-1)
    return x, anc


class SeriesTensors:
    """Series diarias de train en la GPU y armado de lotes (celda, origen) sin materializar las ventanas."""

    def __init__(self, gold: pd.DataFrame, cells: list[str], sigma: float, device="cpu",
                 lookback: int = config.LOOKBACK_DAYS, horizon: int = config.HORIZON_DAYS):
        import torch

        from .cleaning import CALIDAD_VALIDA

        self.cells, self.sigma, self.L, self.H, self.device = cells, sigma, lookback, horizon, device
        self.dates = pd.DatetimeIndex(sorted(gold["fecha"].unique()))
        g = gold.assign(valido=gold["calidad"].isin(CALIDAD_VALIDA))

        def t(col, dtype=torch.float32):
            return torch.tensor(pivot(g, col, cells).reindex(self.dates).to_numpy(dtype=float), dtype=dtype, device=device)

        self.VF, self.V = t("v_filtrada"), t("voltaje")
        self.KA, self.VC = t("kA_z"), t("vcel_z")
        self.VALID = t("valido").bool()
        self.OOS = t("fuera_servicio").bool()
        self.el_flag = torch.tensor([float(c[0] == "B") for c in cells], device=device)
        # Orígenes usables: ventana filtrada completa y sin fuera_servicio (el ancla queda definida)
        vf_ok = pd.DataFrame(np.isfinite(pivot(g, "v_filtrada", cells).reindex(self.dates).to_numpy(float)))
        oos = pd.DataFrame(pivot(g, "fuera_servicio", cells).reindex(self.dates).to_numpy(float))
        ok = (vf_ok.rolling(lookback, min_periods=lookback).min() == 1) & (oos.rolling(lookback, min_periods=1).max() == 0)
        self.ok = torch.tensor(ok.to_numpy(), device=device)
        self.offs_in = torch.arange(-lookback + 1, 1, device=device)
        self.offs_out = torch.arange(1, horizon + 1, device=device)
        self.next_jump = None

    def set_level_shifts(self, saltos: pd.DataFrame) -> None:
        """Registra los saltos de nivel (``fecha``, ``celda``): para cada (t, celda), índice del próximo salto > t.

        Un salto con ``fecha`` k marca el primer día del nuevo nivel; con ``mask_jumps`` se enmascaran los
        objetivos desde k para los orígenes t < k.
        """
        import torch

        T, pos = len(self.dates), {d: i for i, d in enumerate(self.dates)}
        nxt = np.full((T, len(self.cells)), 2 * T, dtype=np.int64)
        for j, c in enumerate(self.cells):
            for k in sorted((pos[d] for d in saltos.loc[saltos["celda"] == c, "fecha"] if d in pos), reverse=True):
                nxt[:k, j] = k
        self.next_jump = torch.tensor(nxt, device=self.device)

    def origins(self, t_min: int, t_max: int, stride: int = 1, phase: int = 0):
        """Pares (celda, t) usables con ``t_min ≤ t ≤ t_max`` y ``(t − phase) % stride == 0``."""
        import torch

        ok = self.ok.clone()
        ok[:t_min] = False
        ok[t_max + 1:] = False
        t_idx = torch.arange(len(self.dates), device=self.device)
        ok &= ((t_idx - phase) % stride == 0)[:, None]
        t, c = torch.nonzero(ok, as_tuple=True)
        return c, t

    def batch(self, c, t, target_limit: int, mask_jumps: bool = False):
        """Lote de entradas y objetivos. Los objetivos posteriores a ``target_limit`` (índice de fecha) se enmascaran.

        Con ``mask_jumps`` también se enmascaran los objetivos a partir del próximo salto de nivel de la celda.
        """
        import torch

        win = t[:, None] + self.offs_in
        cc = c[:, None]
        x, anc = build_inputs(self.VF[win, cc], self.VALID[win, cc], self.KA[win, cc], self.VC[win, cc], self.sigma)
        tgt = t[:, None] + self.offs_out
        dentro = tgt <= min(target_limit, len(self.dates) - 1)
        tgt = tgt.clamp(max=len(self.dates) - 1)
        y = (self.V[tgt, cc] - anc[:, None]) / self.sigma
        mask = dentro & self.VALID[tgt, cc] & torch.isfinite(y)
        if mask_jumps:
            mask &= tgt < self.next_jump[t, c][:, None]
        return x, torch.nan_to_num(y), mask, c, self.el_flag[c]
