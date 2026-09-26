"""Celda NOA-LSTM (sin tanh en la salida), Bi-NOA-LSTM y red multi-horizonte directa de VoltCast."""
from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

from . import config

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # avisos de deprecación de TorchScript

    @torch.jit.script
    def lstm_scan(x_proj: Tensor, w_hh: Tensor, h: Tensor, c: Tensor, noa: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Recorre la secuencia. ``x_proj`` = x·W_ihᵀ + b ya calculado, ``[T, B, 4H]``.

        NOA: ``h_t = o_t ⊙ c_t`` (sin tanh); LSTM estándar: ``h_t = o_t ⊙ tanh(c_t)``.
        """
        outs: list[Tensor] = []
        for t in range(x_proj.size(0)):
            gates = x_proj[t] + torch.mm(h, w_hh.t())
            i, f, g, o = gates.chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * c if noa else torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs), h, c


class NOALSTMLayer(nn.Module):
    """Capa LSTM unidireccional con salida NOA (o estándar si ``noa=False``)."""

    def __init__(self, input_size: int, hidden: int, noa: bool = True):
        super().__init__()
        self.hidden, self.noa = hidden, noa
        self.ih = nn.Linear(input_size, 4 * hidden)
        self.w_hh = nn.Parameter(torch.empty(4 * hidden, hidden))
        nn.init.orthogonal_(self.w_hh)
        with torch.no_grad():  # sesgo de la compuerta de olvido = 1
            self.ih.bias.zero_()
            self.ih.bias[hidden:2 * hidden] = 1.0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: ``[B, T, F]`` → (salidas ``[B, T, H]``, h final ``[B, H]``)."""
        B = x.size(0)
        h0 = x.new_zeros(B, self.hidden)
        out, h, _ = lstm_scan(self.ih(x).transpose(0, 1), self.w_hh, h0, h0.clone(), self.noa)
        return out.transpose(0, 1), h


class BiNOALSTM(nn.Module):
    """Bi-NOA-LSTM de ``layers`` capas. Devuelve la concatenación del estado final de cada dirección."""

    def __init__(self, input_size: int, hidden: int, layers: int = 1, dropout: float = 0.0, noa: bool = True):
        super().__init__()
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        for k in range(layers):
            n_in = input_size if k == 0 else 2 * hidden
            self.fwd.append(NOALSTMLayer(n_in, hidden, noa))
            self.bwd.append(NOALSTMLayer(n_in, hidden, noa))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for k, (f, b) in enumerate(zip(self.fwd, self.bwd)):
            if k:
                x = self.drop(x)
            of, hf = f(x)
            ob, hb = b(x.flip(1))
            x = torch.cat([of, ob.flip(1)], dim=-1)
        return torch.cat([hf, hb], dim=-1)


class VoltCastNet(nn.Module):
    """Canal independiente con pesos compartidos: ventana ``[B, L, F]`` de una celda → ``[B, H]`` (y normalizado).

    Cabezal: concat(h_final, embedding de celda, flag EL) → Linear → GELU → Linear(→ H), sin activación de salida.
    ``use_noa=False`` usa ``nn.LSTM(bidirectional=True)`` (ablación).
    """

    def __init__(self, n_cells: int, n_features: int, hidden: int = 64, layers: int = 1, emb_dim: int = 8,
                 head_hidden: int = 256, horizon: int = config.HORIZON_DAYS, dropout: float = 0.2,
                 use_noa: bool = config.USE_NOA):
        super().__init__()
        self.use_noa = use_noa
        if use_noa:
            self.encoder = BiNOALSTM(n_features, hidden, layers, dropout, noa=True)
        else:
            self.encoder = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                                   dropout=dropout if layers > 1 else 0.0)
        self.emb = nn.Embedding(n_cells, emb_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden + emb_dim + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon),
        )

    def encode(self, x: Tensor) -> Tensor:
        if self.use_noa:
            return self.encoder(x)
        _, (h_n, _) = self.encoder(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)  # última capa: adelante y atrás

    def forward(self, x: Tensor, cell: Tensor, el_flag: Tensor) -> Tensor:
        z = torch.cat([self.encode(x), self.emb(cell), el_flag.unsqueeze(-1)], dim=-1)
        return self.head(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def masked_mse(pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    m = mask.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def run_epoch(model, data, c, t, target_limit: int, batch_size: int, opt=None, clip: float = 1.0,
              generator=None, mask_jumps: bool = False) -> tuple[float, int]:
    """Una pasada sobre los pares (c, t). Con ``opt`` entrena; sin él evalúa. Devuelve (MSE normalizado, n objetivos)."""
    train = opt is not None
    model.train(train)
    n = len(t)
    order = torch.randperm(n, device=t.device, generator=generator) if train else torch.arange(n, device=t.device)
    se, cnt = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x, y, mask, cell, el = data.batch(c[idx], t[idx], target_limit, mask_jumps)
            pred = model(x, cell, el)
            loss = masked_mse(pred, y, mask)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            k = int(mask.sum())
            se += float(loss.detach()) * k
            cnt += k
    return se / max(cnt, 1), cnt


@torch.no_grad()
def rmse_by_horizon(model, data, c, t, target_limit: int, batch_size: int = 4096,
                    mask_jumps: bool = False) -> tuple[Tensor, Tensor]:
    """RMSE por horizonte (en unidades normalizadas) del modelo y de la persistencia (ŷ = 0, es decir, el ancla)."""
    model.eval()
    H = data.H
    se_m = torch.zeros(H, device=t.device)
    se_p = torch.zeros(H, device=t.device)
    n = torch.zeros(H, device=t.device)
    for i in range(0, len(t), batch_size):
        x, y, mask, cell, el = data.batch(c[i:i + batch_size], t[i:i + batch_size], target_limit, mask_jumps)
        m = mask.float()
        se_m += ((model(x, cell, el) - y) ** 2 * m).sum(0)
        se_p += (y ** 2 * m).sum(0)
        n += m.sum(0)
    return (se_m / n).sqrt(), (se_p / n).sqrt()


def fit(model, data, train_range: tuple[int, int], train_target_limit: int, cfg: dict,
        val_pairs=None, val_target_limit: int | None = None, epochs: int | None = None, seed: int = config.SEED,
        mask_jumps: bool = config.MASK_POST_JUMP, log=print) -> dict:
    """Entrena con orígenes submuestreados (``cfg['stride']`` días, fase aleatoria en cada época).

    Con ``val_pairs`` aplica early stopping (``cfg['patience']``) y restaura los mejores pesos;
    sin ellos entrena exactamente ``epochs`` épocas (reentrenamiento final).
    """
    import copy
    import time

    gen = torch.Generator(device=data.device).manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    max_ep = epochs or cfg["max_epochs"]
    hist, best, best_state, best_ep, wait = [], float("inf"), None, 0, 0
    for ep in range(1, max_ep + 1):
        t0 = time.time()
        phase = int(torch.randint(cfg["stride"], (1,), generator=rng))
        c, t = data.origins(*train_range, stride=cfg["stride"], phase=phase)
        tr_mse, _ = run_epoch(model, data, c, t, train_target_limit, cfg["batch_size"], opt, cfg["clip"], gen, mask_jumps)
        row = {"epoca": ep, "train_mse": tr_mse, "segundos": time.time() - t0}
        if val_pairs is not None:
            va_mse, _ = run_epoch(model, data, *val_pairs, val_target_limit, 4 * cfg["batch_size"], mask_jumps=mask_jumps)
            row["val_mse"] = va_mse
            if va_mse < best - 1e-5:
                best, best_ep, wait = va_mse, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
        hist.append(row)
        if log and (ep == 1 or ep % 5 == 0 or wait == 0):
            log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        if val_pairs is not None and wait >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"historia": hist, "mejor_epoca": best_ep or max_ep, "mejor_val_mse": best if best_state else None}


# --------------------------------------------------------------------------- #
# Persistencia del modelo
# --------------------------------------------------------------------------- #
MODEL_KWARGS = ["hidden", "layers", "emb_dim", "head_hidden", "dropout"]
VERSION = "voltcast-poc-0.1"


def build_model(n_cells: int, n_features: int, cfg: dict, use_noa: bool) -> VoltCastNet:
    return VoltCastNet(n_cells, n_features, **{k: cfg[k] for k in MODEL_KWARGS}, use_noa=use_noa)


def save_checkpoint(path, model: VoltCastNet, meta: dict) -> None:
    """Guarda pesos + todo lo necesario para reconstruir e interpretar el modelo (config, escaladores, celdas…)."""
    torch.save({"state_dict": model.state_dict(), "version": VERSION, "torch": torch.__version__, **meta}, path)


def load_checkpoint(path, device="cpu") -> tuple[VoltCastNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ck["cells"]), len(ck["features"]), ck["train_cfg"], ck["use_noa"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_volts(model: VoltCastNet, vf_win, valid_win, ka_win, vc_win, sigma: float, cell_idx, el_flag,
                  device="cpu"):
    """Pronóstico en voltios para N celdas de un origen en un único forward pass.

    Entradas ``[N, L]`` (numpy); devuelve (``V̂ [N, H]`` = ancla + σ_Δ · ŷ, ancla ``[N]``).
    """
    import numpy as np

    from .dataset import build_inputs

    def t(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    x, anc = build_inputs(t(np.nan_to_num(vf_win)), t(valid_win, torch.bool), t(ka_win), t(vc_win), sigma)
    y = model.eval()(x, t(cell_idx, torch.long), t(el_flag))
    return (anc[:, None] + sigma * y).cpu().numpy(), anc.cpu().numpy()
