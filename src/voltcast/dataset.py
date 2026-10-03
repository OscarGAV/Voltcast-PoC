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


def next_jump_index(dates: pd.DatetimeIndex, cells: list[str], saltos: pd.DataFrame) -> np.ndarray:
    """``[T, C]``: índice (en ``dates``) del próximo salto de nivel posterior a cada fecha; ``2T`` si no hay."""
    T, pos = len(dates), {d: i for i, d in enumerate(dates)}
    nxt = np.full((T, len(cells)), 2 * T, dtype=np.int64)
    for j, c in enumerate(cells):
        for k in sorted((pos[d] for d in saltos.loc[saltos["celda"] == c, "fecha"] if d in pos), reverse=True):
            nxt[:k, j] = k
    return nxt


# --------------------------------------------------------------------------- #
# Tensores para el modelo (todo cabe en memoria de la GPU: ~2000 días × 2 × N celdas)
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

        self.next_jump = torch.tensor(next_jump_index(self.dates, self.cells, saltos), device=self.device)

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


class MatrixTensors:
    """Lotes (grupo, origen) para los modelos que pronostican todas las celdas a la vez (A MIMO, C grafo, D TCN).

    ``X[g]``: entradas diarias ``[T, ...]`` (ya normalizadas, sin NaN); ``Y[g]``: objetivos ``[T, N]`` normalizados
    (NaN = no evaluable); ``VALID[g]``: ``[T, N]``. Misma interfaz que ``SeriesTensors`` (``origins``, ``batch``),
    así que se entrenan con ``model.fit``; ``batch`` devuelve el índice de grupo en lugar del de celda.
    """

    def __init__(self, X: dict, Y: dict, VALID: dict, dates: pd.DatetimeIndex, device="cpu",
                 lookback: int = config.LOOKBACK_DAYS, horizon: int = config.HORIZON_DAYS):
        import torch

        self.groups = list(X)
        self.dates, self.L, self.H, self.device = dates, lookback, horizon, device

        def st(d, dtype=torch.float32):
            return torch.stack([torch.as_tensor(np.asarray(d[g]), dtype=dtype) for g in self.groups]).to(device)

        self.X, self.Y, self.VALID = st(X), st(Y), st(VALID, torch.bool)
        self.offs_in = torch.arange(-lookback + 1, 1, device=device)
        self.offs_out = torch.arange(1, horizon + 1, device=device)

    def origins(self, t_min: int, t_max: int, stride: int = 1, phase: int = 0):
        import torch

        t = torch.arange(max(t_min, self.L - 1), t_max + 1, device=self.device)
        t = t[(t - phase) % stride == 0]
        g = torch.arange(len(self.groups), device=self.device).repeat_interleave(len(t))
        return g, t.repeat(len(self.groups))

    def batch(self, c, t, target_limit: int, mask_jumps: bool = False):
        import torch

        win = t[:, None] + self.offs_in
        x = self.X[c[:, None], win]
        tgt = t[:, None] + self.offs_out
        dentro = tgt <= min(target_limit, len(self.dates) - 1)
        tgt = tgt.clamp(max=len(self.dates) - 1)
        y = self.Y[c[:, None], tgt]  # [B, H, N]
        mask = dentro[..., None] & self.VALID[c[:, None], tgt] & torch.isfinite(y)
        return x, torch.nan_to_num(y), mask, c, c.float()
