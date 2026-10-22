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
LOOKBACK_DAYS = 30  # L de los modelos del Benchmarking (especificación; docs/hiperparametros.md 6.3)
LOOKBACK_PROPIO = 5  # L de la Bi-LSTM propia (prueba de menor error, docs/hiperparametros.md 6.4)
HORIZON_DAYS = 180  # H máx.; se evalúan cortes a 60 y 180
EVAL_HORIZONS = [60, 180]
VAL_FRACTION = 0.20
TARGET_CELLS = None  # None = todas las celdas detectadas en el Excel; o lista, p. ej. ["A_c001", "B_c017"]
# Celdas físicas por electrolizador. [Volts] suma todas las celdas físicas aunque el Excel incluya menos;
# se usa para que la covariable V_total / celdas en servicio no se distorsione. None = usar las del Excel.
N_CELDAS_FISICAS = 181
ANCHOR_DAYS = 3  # días promediados para el ancla del nivel (prueba de menor error, docs/hiperparametros.md 6.4)
KA_SHUTDOWN = 10.0  # kA por debajo = día de paro (prueba de menor error frente a 13,5 kA, docs/hiperparametros.md 6.2)
MAX_INTERP_DAYS = 12  # huecos ≤ 12 días se interpolan (prueba de menor error, docs/hiperparametros.md 6.2)
OOS_MIN_DAYS = 7  # duración mínima de un tramo fuera_servicio
OOS_MERGE_DAYS = 5  # interrupciones ≤ 5 días no cortan un tramo fuera_servicio
STEP_THRESHOLD_V = 0.10  # salto de nivel (V) para el diagnóstico de la evaluación (docs/hiperparametros.md 6.4)
STEP_THRESHOLD_MASK = 0.03  # salto de nivel (V) para la máscara de entrenamiento (prueba de menor error)
STEP_WINDOW_DAYS = 7  # días de mediana antes/después para detectar un salto
BO_TRIALS = 30  # evaluaciones de la BO-VMD por celda (Makri et al., 2026)
EVAL_STRIDE = 7  # días entre orígenes de validación (rolling-origin)
VMD_HISTORY = 365  # días de historia usados para VMD causal en validación/inferencia
VMD_MIN_LEN = 60  # segmento en servicio mínimo (días) para aplicar VMD
N_JOBS = -1  # procesos de joblib (-1 = todos los núcleos)
COV_CLIP = 5.0  # |z| máximo de las covariables estandarizadas
USE_NOA = True  # True = Bi-NOA-LSTM; False = nn.LSTM estándar (ablación)
MASK_POST_JUMP = True  # enmascarar en el loss los objetivos posteriores a un salto de nivel (histórico de train)
ES_BLOCK_DAYS = 270  # último bloque de train usado para early stopping
REFIT_FULL_TRAIN = True  # tras el early stopping, reentrenar con todo train durante las épocas elegidas
TRAIN = {
    "hidden": 64, "layers": 1, "emb_dim": 8, "head_hidden": 256, "dropout": 0.2,
    "batch_size": 1024, "lr": 1e-3, "weight_decay": 1e-4, "clip": None,
    "max_epochs": 200, "patience": 20, "stride": 5,  # paciencia y máximo: Khan et al. (2026); orígenes cada 5 días
}
SEED = 42

# Propuesta B: Seq2Seq BiLSTM + atención multi-cabeza (Gou et al., 2024) + fusión con Kalman (Khan et al., 2026)
SEQ2SEQ = {
    "arch": "seq2seq", "hidden": 32, "dec_hidden": 64,  # encoder / decoder: Gou et al. (2024)
    "heads": 4, "teacher_forcing": False,  # pendientes de la prueba de menor error (docs/hiperparametros.md 10.4.1)
    "emb_dim": 8, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "clip": None,  # Khan (Tabla 7)
    "batch_size": 512,  # lote limitado por la memoria de la GPU (8 GB)
    "max_epochs": 200, "patience": 20, "stride": 10,  # Khan; decoder de 180 pasos: orígenes cada 10 días
}
ENSEMBLE_SEEDS = [42, 43, 44]
PROPIO_SEEDS = [42, 43, 44, 45, 46]  # ensamble de semillas de la Bi-LSTM v2 (notebook 06e)
FUSION_ALPHAS = [round(0.30 + 0.01 * i, 2) for i in range(41)]  # α ∈ [0,3; 0,7] con paso 0,01 (Khan et al., 2026, ec. 16)

# Propuesta A (Makri et al., 2026): modos BO-VMD + RFR-SHAP + Min-Max + Bi-NOA-LSTM MIMO
# Óptimo que Makri obtuvo con SOMA (Tabla 6): 3 capas de 51 unidades, dropout 0, lr 3e-3, lote 16, 150 épocas, Adam
MIMO_TRAIN = {"arch": "mimo", "layers": 3, "hidden": 51, "dropout": 0.0, "lr": 3e-3, "weight_decay": 0.0, "clip": None,
              "batch_size": 16, "stride": 1, "max_epochs": 150}
RFR_TREES = 100  # árboles del Random Forest de RFR-SHAP (Makri, sec. 3.3); el resto, valores por defecto de scikit-learn
RFR_TOP = 12  # variables elegidas por SHAP (Makri, sec. 2.2 y 3.3)

# Propuesta C: ST-GRU (Palan et al., Tablas 3 y 4; lote de 64, Adam con weight decay, early stopping)
STGRU = {"arch": "stgru", "hidden": 512, "layers": 2, "dropout": 0.2, "lr": 7.69e-4,
         "weight_decay": 1e-4,  # Palan no publica el valor: 1e-4, la L2 de Khan y el máximo del rango que Palan busca para otro modelo
         "clip": None, "batch_size": 64, "micro_batch": 16,  # lote de 64 en 4 partes (mismo gradiente; 8 GB de GPU)
         "stride": 1, "max_epochs": 90, "patience": 20}
# Propuesta D: TCN multi-salida (Duan: lr 1e-3, lote 32; Palan, Tabla 4: 64 canales, kernel 3, 4 capas, dropout 0,2)
TCN = {"arch": "tcn", "channels": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 0.0, "clip": None,
       "batch_size": 32, "stride": 1, "max_epochs": 200, "patience": 20}

# Umbrales de calidad usados en EDA y Silver
V_OUT_OF_SERVICE = 0.5  # V por debajo (o negativo) con kA normal = celda fuera de servicio / puenteada
V_MAX_PLAUSIBLE = 5.0  # V por encima = atípico


def ensure_dirs() -> None:
    """Crea las carpetas de datos, modelos y reportes si no existen."""
    for d in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(filename: str | None = None) -> Path:
    return RAW_DIR / (filename or RAW_FILE)
