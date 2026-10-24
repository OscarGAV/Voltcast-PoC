"""Celda NOA-LSTM (compuerta de salida sin sigmoide, Makri et al., 2026), Bi-NOA-LSTM y red multi-horizonte directa."""
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

        NOA (Makri et al., 2026, ec. 22): la compuerta de salida es lineal, sin sigmoide, y ``h_t = o_t ⊙ tanh(c_t)``;
        LSTM estándar: ``h_t = σ(o_t) ⊙ tanh(c_t)``.
        """
        outs: list[Tensor] = []
        for t in range(x_proj.size(0)):
            gates = x_proj[t] + torch.mm(h, w_hh.t())
            i, f, g, o = gates.chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = (o if noa else torch.sigmoid(o)) * torch.tanh(c)
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
    """Bi-NOA-LSTM de ``layers`` capas. Combina las dos direcciones por concatenación (``merge="concat"``) o por suma
    (``merge="sum"``, como la Figura 4 de Makri et al., 2026) y devuelve la combinación del estado final."""

    def __init__(self, input_size: int, hidden: int, layers: int = 1, dropout: float = 0.0, noa: bool = True,
                 merge: str = "concat"):
        super().__init__()
        self.merge = merge
        self.out_dim = hidden if merge == "sum" else 2 * hidden
        self.fwd = nn.ModuleList()
        self.bwd = nn.ModuleList()
        for k in range(layers):
            n_in = input_size if k == 0 else self.out_dim
            self.fwd.append(NOALSTMLayer(n_in, hidden, noa))
            self.bwd.append(NOALSTMLayer(n_in, hidden, noa))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for k, (f, b) in enumerate(zip(self.fwd, self.bwd)):
            if k:
                x = self.drop(x)
            of, hf = f(x)
            ob, hb = b(x.flip(1))
            x = of + ob.flip(1) if self.merge == "sum" else torch.cat([of, ob.flip(1)], dim=-1)
        return hf + hb if self.merge == "sum" else torch.cat([hf, hb], dim=-1)


class VoltCastNet(nn.Module):
    """Canal independiente con pesos compartidos: ventana ``[B, L, F]`` de una celda → ``[B, H]`` (y normalizado).

    Cabezal: concat(h_final, embedding de celda, flag EL) → Linear → ReLU → Linear(→ H), sin activación de salida.
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
            nn.ReLU(),  # ReLU en las capas densas, como Song et al. (2025) y Gou et al. (2024)
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


class Seq2SeqAttention(nn.Module):
    """Propuesta B: encoder BiLSTM + decoder LSTM con atención multi-cabeza, **autorregresivo paso a paso**.

    El decoder genera ``ŷ_{t0+1} … ŷ_{t0+H}`` realimentando su propia predicción, así que el error puede acumularse
    con el horizonte (deriva por recursión), a diferencia de ``VoltCastNet``. Con ``teacher_forcing``, en
    entrenamiento cada paso recibe el valor real anterior (donde no hay valor medido, la propia predicción).
    """

    def __init__(self, n_cells: int, n_features: int, hidden: int = 64, dec_hidden: int = 128, heads: int = 4,
                 emb_dim: int = 8, dropout: float = 0.2, horizon: int = config.HORIZON_DAYS,
                 teacher_forcing: bool = False):
        super().__init__()
        d = 2 * hidden
        self.horizon, self.teacher_forcing = horizon, teacher_forcing
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(n_cells, emb_dim)
        self.init_h = nn.Linear(d + emb_dim + 1, dec_hidden)
        self.init_c = nn.Linear(d + emb_dim + 1, dec_hidden)
        self.heads, self.dh = heads, d // heads
        # Atención multi-cabeza: claves y valores del encoder se proyectan una sola vez (no en cada paso)
        self.q = nn.Linear(dec_hidden, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.decoder = nn.LSTMCell(1 + d, dec_hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dec_hidden + d, 1)  # una capa totalmente conectada sobre [estado, contexto]: Gou et al. (2024), ec. 14

    def forward(self, x: Tensor, cell: Tensor, el_flag: Tensor, y_real: Tensor | None = None,
                mask: Tensor | None = None) -> Tensor:
        enc, (h_n, _) = self.encoder(x)  # [B, L, d]
        z = self.drop(torch.cat([h_n[-2], h_n[-1], self.emb(cell), el_flag.unsqueeze(-1)], dim=-1))
        h, c = torch.tanh(self.init_h(z)), self.init_c(z)
        B, L, d = enc.shape
        K = self.k(enc).view(B, L, self.heads, self.dh).transpose(1, 2)  # [B, heads, L, dh]
        V = self.v(enc).view(B, L, self.heads, self.dh).transpose(1, 2)
        escala = self.dh ** -0.5
        y = x[:, -1, :1]  # último voltaje relativo al ancla
        outs = []
        for k in range(self.horizon):
            q = self.q(h).view(B, self.heads, 1, self.dh)
            w = torch.softmax(q @ K.transpose(-1, -2) * escala, dim=-1)  # [B, heads, 1, L]
            ctx = self.o((w @ V).reshape(B, d))
            h, c = self.decoder(torch.cat([y, ctx], dim=-1), (h, c))
            y = self.out(self.drop(torch.cat([h, ctx], dim=-1)))
            outs.append(y)
            if y_real is not None:  # teacher forcing: el paso siguiente recibe el valor real
                y = torch.where(mask[:, k:k + 1], y_real[:, k:k + 1], y)
        return torch.cat(outs, dim=1)


class MIMOBiNOALSTM(nn.Module):
    """Propuesta A (benchmarking): Bi-NOA-LSTM **MIMO**. Matriz de entrada ``[B, L, n_in]`` (todas las celdas +
    covariables) → matriz de salida ``[B, H, n_out]`` (todas las celdas) en un solo pase.

    Proyección final densa sin activación (NOA en la celda y en la salida).
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 128, layers: int = 1, dropout: float = 0.2,
                 horizon: int = config.HORIZON_DAYS, use_noa: bool = True, merge: str = "sum"):
        super().__init__()
        self.n_out, self.horizon = n_out, horizon
        self.encoder = BiNOALSTM(n_in, hidden, layers, dropout, noa=use_noa, merge=merge)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.encoder.out_dim, horizon * n_out)

    def forward(self, x: Tensor, cell: Tensor | None = None, el_flag: Tensor | None = None) -> Tensor:
        return self.proj(self.drop(self.encoder(x))).view(-1, self.horizon, self.n_out)


class STGRU(nn.Module):
    """Propuesta C: ST-GRU (Palan et al.). En cada día, una **GRU espacial** recorre las variables de ese día como una
    secuencia (las celdas y kA) y su último estado es la representación del día; una **GRU temporal** recorre esas
    representaciones de los ``L`` días; su último estado pasa por ReLU y una capa densa que proyecta ``[B, H, n_out]``.

    Entrada ``[B, L, F]``; las dos GRU tienen ``hidden`` unidades y ``layers`` capas (Palan, Tablas 3 y 4).
    """

    def __init__(self, n_in: int, n_out: int, hidden: int = 512, layers: int = 2, dropout: float = 0.2,
                 horizon: int = config.HORIZON_DAYS):
        super().__init__()
        self.n_out, self.horizon = n_out, horizon
        self.spatial = nn.GRU(1, hidden, layers, batch_first=True, dropout=dropout)
        self.temporal = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon * n_out)

    def forward(self, x: Tensor, cell: Tensor | None = None, el_flag: Tensor | None = None) -> Tensor:
        B, L, F = x.shape
        _, h = self.spatial(x.reshape(B * L, F, 1))  # cada día: secuencia de sus F variables
        _, h = self.temporal(h[-1].view(B, L, -1))
        return self.head(torch.relu(h[-1])).view(B, self.horizon, self.n_out)


class TCNBlock(nn.Module):
    """Bloque residual del TCN: dos convoluciones causales dilatadas con *weight normalization*, ReLU y dropout (Duan;
    TCN genérico de Bai et al., 2018); si cambia el número de canales, la conexión residual usa una convolución 1×1."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation  # relleno solo a la izquierda: convolución causal
        wn = nn.utils.parametrizations.weight_norm
        self.c1 = wn(nn.Conv1d(c_in, c_out, kernel, dilation=dilation))
        self.c2 = wn(nn.Conv1d(c_out, c_out, kernel, dilation=dilation))
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(self.down(x) + y)  # conexión residual


class TCNMultiOutput(nn.Module):
    """Propuesta D: TCN multi-salida. Canales de entrada = celdas + covariables ``[B, L, n_in]``; bloques residuales
    causales dilatados (dilatación 2^k en el bloque k) con ``channels`` filtros por bloque y una capa densa que proyecta
    en paralelo ``[B, H, n_out]``. Duan et al. (2026): 3 capas de 32, 64 y 128 filtros, kernel 3.
    """

    def __init__(self, n_in: int, n_out: int, channels=(32, 64, 128), kernel: int = 3, dropout: float = 0.2,
                 horizon: int = config.HORIZON_DAYS):
        super().__init__()
        self.n_out, self.horizon = n_out, horizon
        canales = [channels] * 4 if isinstance(channels, int) else list(channels)
        self.blocks = nn.Sequential(*[TCNBlock(n_in if i == 0 else canales[i - 1], c, kernel, 2 ** i, dropout)
                                      for i, c in enumerate(canales)])
        self.head = nn.Linear(canales[-1], horizon * n_out)

    def forward(self, x: Tensor, cell: Tensor | None = None, el_flag: Tensor | None = None) -> Tensor:
        z = self.blocks(x.transpose(1, 2))[:, :, -1]  # representación en t0
        return self.head(z).view(-1, self.horizon, self.n_out)


class GroupModels(nn.Module):
    """Un modelo por grupo (p. ej. por electrolizador). Todas las muestras de un lote deben ser del mismo grupo."""

    def __init__(self, models: dict):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.keys = list(models)

    def forward(self, x: Tensor, cell: Tensor, el_flag: Tensor | None = None) -> Tensor:
        return self.models[self.keys[int(cell[0])]](x, cell, el_flag)


class Ensemble(nn.Module):
    """Promedio de las salidas de varios modelos con la misma interfaz ``(x, cell, el_flag) → [B, H]``."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor, cell: Tensor, el_flag: Tensor) -> Tensor:
        return torch.stack([m(x, cell, el_flag) for m in self.models]).mean(0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def masked_mse(pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    m = mask.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def run_epoch(model, data, c, t, target_limit: int, batch_size: int, opt=None, clip: float | None = None,
              generator=None, mask_jumps: bool = False, micro_batch: int | None = None) -> tuple[float, int]:
    """Una pasada sobre los pares (c, t). Con ``opt`` entrena; sin él evalúa. Devuelve (MSE normalizado, n objetivos).

    Con ``micro_batch``, cada lote se procesa en partes y se acumulan los gradientes, ponderando cada parte por sus
    objetivos: el gradiente es el del lote completo, con menos memoria de GPU.
    """
    train = opt is not None
    model.train(train)
    n = len(t)
    order = torch.randperm(n, device=t.device, generator=generator) if train else torch.arange(n, device=t.device)
    se, cnt = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch_size):
            partes = [data.batch(c[j], t[j], target_limit, mask_jumps)
                      for j in order[i:i + batch_size].split(micro_batch or batch_size)]
            k_lote = sum(int(p[2].sum()) for p in partes)
            if train:
                opt.zero_grad(set_to_none=True)
            for x, y, mask, cell, el in partes:
                tf = train and getattr(model, "teacher_forcing", False)
                pred = model(x, cell, el, y, mask) if tf else model(x, cell, el)
                loss = masked_mse(pred, y, mask)
                k = int(mask.sum())
                if train:
                    (loss * k / max(k_lote, 1)).backward()
                se += float(loss.detach()) * k
                cnt += k
            if train:
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
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
    # Adam como todas las fuentes; weight_decay en Adam es la penalización L2 (equivale a la L2 de Khan et al.)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    max_ep = epochs or cfg["max_epochs"]
    hist, best, best_state, best_ep, wait = [], float("inf"), None, 0, 0
    for ep in range(1, max_ep + 1):
        t0 = time.time()
        phase = int(torch.randint(cfg["stride"], (1,), generator=rng))
        c, t = data.origins(*train_range, stride=cfg["stride"], phase=phase)
        tr_mse, _ = run_epoch(model, data, c, t, train_target_limit, cfg["batch_size"], opt, cfg.get("clip"), gen, mask_jumps,
                              cfg.get("micro_batch"))
        row = {"epoca": ep, "train_mse": tr_mse, "segundos": time.time() - t0}
        if val_pairs is not None:
            va_mse, _ = run_epoch(model, data, *val_pairs, val_target_limit, 4 * cfg.get("micro_batch", cfg["batch_size"]),
                                  mask_jumps=mask_jumps)
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
SEQ2SEQ_KWARGS = ["hidden", "dec_hidden", "heads", "emb_dim", "dropout"]
VERSION = "voltcast-poc-0.1"


def build_model(n_cells: int, n_features: int, cfg: dict, use_noa: bool = True) -> nn.Module:
    """Construye la red según ``cfg['arch']``.

    ``seq2seq`` → ``Seq2SeqAttention``; ``mimo`` → ``MIMOBiNOALSTM``; ``tcn`` → ``TCNMultiOutput``;
    ``stgru`` → ``STGRU`` (necesita ``cfg['adjacency']``); sin ``arch`` → ``VoltCastNet`` (Bi-NOA-LSTM o Bi-LSTM).
    Con ``cfg['groups']``, un modelo por grupo (``GroupModels``).
    """
    arch = cfg.get("arch")
    if cfg.get("groups"):
        sub = {k: v for k, v in cfg.items() if k != "groups"}
        return GroupModels({g: build_model(n_cells, n_features, sub, use_noa) for g in cfg["groups"]})
    if arch == "seq2seq":
        return Seq2SeqAttention(n_cells, n_features, **{k: cfg[k] for k in SEQ2SEQ_KWARGS},
                                teacher_forcing=cfg.get("teacher_forcing", False))
    if arch == "mimo":
        return MIMOBiNOALSTM(cfg["n_in"], cfg["n_out"], cfg["hidden"], cfg["layers"], cfg["dropout"],
                             merge=cfg.get("merge", "sum"))
    if arch == "tcn":
        return TCNMultiOutput(cfg["n_in"], cfg["n_out"], cfg["channels"], kernel=cfg.get("kernel", 3),
                              dropout=cfg["dropout"])
    if arch == "stgru":
        return STGRU(cfg["n_in"], cfg["n_out"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    return VoltCastNet(n_cells, n_features, **{k: cfg[k] for k in MODEL_KWARGS}, use_noa=use_noa)


def save_checkpoint(path, model: nn.Module, meta: dict) -> None:
    """Guarda pesos + todo lo necesario para reconstruir e interpretar el modelo (config, escaladores, celdas…)."""
    torch.save({"state_dict": model.state_dict(), "version": VERSION, "torch": torch.__version__, **meta}, path)


def load_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    def uno():
        return build_model(len(ck["cells"]), len(ck["features"]), ck["train_cfg"], ck.get("use_noa", True))

    model = Ensemble([uno() for _ in range(ck["n_modelos"])]) if ck.get("n_modelos") else uno()  # ensamble de semillas
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_volts(model: nn.Module, vf_win, valid_win, ka_win, vc_win, sigma: float, cell_idx, el_flag,
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
