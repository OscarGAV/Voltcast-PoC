"""Rutas y parámetros centrales de la PoC VoltCast.

Detecta si corre en Google Colab (monta Drive y ajusta ``BASE_DIR``) o en local.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Entorno
# --------------------------------------------------------------------------- #
try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Carpeta del proyecto dentro de Google Drive (solo Colab). Se puede cambiar con
# la variable de entorno VOLTCAST_DRIVE_DIR.
COLAB_DRIVE_DIR = os.environ.get("VOLTCAST_DRIVE_DIR", "/content/drive/MyDrive/Poc-TP1")


def _resolve_base_dir() -> Path:
    if "VOLTCAST_BASE_DIR" in os.environ:
        return Path(os.environ["VOLTCAST_BASE_DIR"]).resolve()
    if IN_COLAB:
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        return Path(COLAB_DRIVE_DIR)
    # src/voltcast/config.py -> raíz del proyecto
    return Path(__file__).resolve().parents[2]


BASE_DIR = _resolve_base_dir()

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
EDA_DIR = REPORTS_DIR / "eda"

# --------------------------------------------------------------------------- #
# Datos de origen
# --------------------------------------------------------------------------- #
RAW_FILE = "VoltajesDiariosDesde2020.xlsx"  # o "Dataset Sintetizado.xlsx" para pruebas rápidas
SHEETS = {"A": "EL A 2020_2026", "B": "EL B 2020_2026"}  # electrolizador -> hoja

# --------------------------------------------------------------------------- #
# Parámetros de la PoC
# --------------------------------------------------------------------------- #
LOOKBACK_DAYS = 30  # L
HORIZON_DAYS = 180  # H máx.; se evalúan cortes a 60 y 180
EVAL_HORIZONS = [60, 180]
VAL_FRACTION = 0.20
TARGET_CELLS = None  # None = todas (2 × 181); o lista, p. ej. ["A_c001", "B_c017"]
ANCHOR_DAYS = 7  # días promediados para el ancla del nivel
KA_SHUTDOWN = 10.0  # kA por debajo = día de paro
MAX_INTERP_DAYS = 3  # huecos ≤ 3 días se interpolan
BO_SAMPLE_CELLS = 20  # celdas para BO-VMD (10 por electrolizador, muestreo estratificado)
BO_TRIALS = 30  # evaluaciones de la optimización bayesiana por celda de la muestra
EVAL_STRIDE = 7  # días entre orígenes de validación (rolling-origin)
VMD_HISTORY = 365  # días de historia usados para VMD causal en validación/inferencia
USE_NOA = True  # True = Bi-NOA-LSTM; False = nn.LSTM estándar (ablación)
SEED = 42

# Umbrales de calidad usados en EDA y Silver
V_OUT_OF_SERVICE = 0.5  # V por debajo (o negativo) con kA normal = celda fuera de servicio / puenteada
V_MAX_PLAUSIBLE = 5.0  # V por encima = atípico


def ensure_dirs() -> None:
    """Crea las carpetas de datos, modelos y reportes si no existen."""
    for d in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(filename: str | None = None) -> Path:
    return RAW_DIR / (filename or RAW_FILE)
