# VoltCast — PoC de pronóstico de voltaje por celda

Prueba de Concepto para pronosticar el **voltaje de cada celda** de los electrolizadores EL A y EL B (150 celdas por electrolizador en el dataset actual) con horizontes de **1 a 180 días**, comparando **nuestro modelo** con los 4 modelos mejor puntuados en un Benchmarking de la literatura y con 2 métodos de referencia.

El proyecto está organizado como una serie de notebooks según **CRISP-DM**, con los datos en **arquitectura Medallion** (Bronze → Silver → Gold). La especificación completa está en [`prompt.md`](prompt.md).

## Resultado principal

Evaluación sobre 486 días de validación (abr-2025 → ago-2026), 70 fechas de origen y ~2,9 millones de pronósticos, todos los modelos sobre los mismos objetivos y contra el voltaje real sin filtrar:

| Puesto | Modelo | RMSE 1–180 d | MAE 1–180 d | RMSE 180 d |
|---|---|---|---|---|
| 1 | **Propio · Bi-LSTM** (modelo elegido) | **0,0585 V** | **0,0270 V** | 0,0691 V |
| 2 | Propio · Bi-NOA-LSTM | 0,0601 V | 0,0274 V | **0,0689 V** |
| 3 | Persistencia (referencia) | 0,0602 V | 0,0334 V | 0,0787 V |
| 4 | B · Seq2Seq BiLSTM + atención + Kalman (Benchmarking 4.º) | 0,0744 V | 0,0428 V | 0,1037 V |
| 5 | C · ST-GRU (Benchmarking 2.º) | 0,0879 V | 0,0604 V | 0,0977 V |
| 6 | A2 · BO-VMD + SOMA-Bi-NOA-LSTM, MIMO conjunto (Benchmarking 1.º) | 0,1143 V | 0,0892 V | 0,1112 V |
| 7 | Tendencia lineal (referencia) | 0,1179 V | 0,0473 V | 0,1989 V |
| 8 | A1 · BO-VMD + SOMA-Bi-NOA-LSTM, MIMO por electrolizador (Benchmarking 1.º) | 0,1186 V | 0,0892 V | 0,1130 V |
| 9 | D · TCN Multi-Output (Benchmarking 3.º) | 0,1487 V | 0,1038 V | 0,1651 V |

- Nuestras variantes son las mejores en todas las métricas sobre el horizonte completo y, a 180 días, superan al mejor modelo del Benchmarking con significancia estadística. A 60 días la ventaja no es significativa y la persistencia queda levemente mejor en RMSE.
- Ningún modelo del Benchmarking supera a la persistencia; el ranking empírico (B > C > A > D) no coincide con el de la literatura (A > C > D > B).
- **Límite:** el R² sobre el cambio de voltaje es ≈ 0 para todos los modelos. La ventaja de nuestro modelo es seguir mejor el nivel y la deriva de cada celda; ninguno anticipa los cambios individuales, dominados por las intervenciones de mantenimiento.

Detalle, figuras y discusión: notebook [`07_evaluacion.ipynb`](notebooks/07_evaluacion.ipynb) y [`reports/metricas_comparativas.csv`](reports/metricas_comparativas.csv).

## Instalación

Requisitos: Windows o Linux, **Python 3.13** y, opcionalmente, una GPU NVIDIA con CUDA 12.8 (los notebooks también funcionan en CPU, más lento).

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name voltcast-venv --display-name "VoltCast (.venv 3.13)"
.venv\Scripts\nbstripout --install
```

- `requirements.txt` instala PyTorch con CUDA 12.8. Para solo CPU, quitar la línea `--extra-index-url`.
- `nbstripout` borra las salidas de los notebooks al hacer commit, para no subir gráficos ni datos.
- En Windows con el **Control inteligente de aplicaciones** activado, Windows puede bloquear DLL de pandas (`DLL load failed ... Control de aplicaciones`).

## Datos

Los datos son **confidenciales** y no están en el repositorio (`data/`, `*.xlsx` y `*.parquet` están en `.gitignore`).

1. Copiar el Excel de origen a `data/raw/VoltajesDiariosDesde2020.xlsx` (nombre configurable en `RAW_FILE` de [`config.py`](src/voltcast/config.py)).
2. Estructura esperada: hojas `EL A 2020_2026` y `EL B 2020_2026`, columnas `Date`, `[Volts]`, `[kA]` y una columna numerada por celda. La ingesta **detecta la cantidad de celdas**; si cambia o se renumera, hay que volver a ejecutar todo y reentrenar.
3. `[Volts]` suma las 181 celdas físicas aunque el Excel incluya menos (`N_CELDAS_FISICAS` en `config.py`).

## Ejecución

Abrir los notebooks con el kernel **VoltCast (.venv 3.13)** y ejecutarlos en orden. Cada uno lee solo la capa anterior y escribe solo la suya, así que se puede reejecutar desde cualquier punto.

| Notebook | Fase CRISP-DM | Qué hace | Tiempo aprox. (RTX 4060) |
|---|---|---|---|
| `01_comprension_negocio` | 1 | Contexto, objetivos, criterios de éxito y alcance | — |
| `02_ingesta_bronze` | 2 | Excel → Bronze sin transformar valores; valida el esquema | 10 s |
| `03_comprension_datos` | 2 | Análisis exploratorio y reglas de calidad | 25 s |
| `04_preparacion_silver` | 3 | Limpieza, máscara de calidad, saltos de nivel, split 80/20 | 15 s |
| `05_preparacion_gold_bovmd` | 3 | BO-VMD (optimización bayesiana), filtrado causal, ancla y covariables | 2 min |
| `05b_preparacion_gold_kalman` | 3 | Filtro de Kalman (para la Propuesta B) | 1 min |
| `06_modelado` | 4 | **Nuestras variantes:** Bi-NOA-LSTM y Bi-LSTM | 8 min |
| `06b_modelado_seq2seq_kalman` | 4 | Propuesta B: Seq2Seq + atención + ensamble con Kalman | 14 min |
| `06c_modelado_propuesta_a` | 4 | Propuesta A (A1 y A2): RFR-SHAP + SOMA + Bi-NOA-LSTM MIMO | ~3 h (SOMA) |
| `06d_modelado_stgru_tcn` | 4 | Propuestas C (ST-GRU) y D (TCN) | 8 min |
| `07_evaluacion` | 5 | Comparación de los 9 modelos, ranking y conclusiones | 1 min |

Los notebooks también pueden ejecutarse en **Google Colab**: detectan Colab, montan Google Drive y buscan el proyecto en `MyDrive/Poc-TP1`.

## Estructura

```
Poc-TP1/
├── docs/                  # propuestas de modelos (Benchmarking), historias de usuario, prompt original
├── notebooks/             # 01–07 (CRISP-DM)
├── src/voltcast/          # código compartido
│   ├── config.py          # rutas y parámetros
│   ├── io.py              # lectura del Excel, capas Parquet con _metadata.json
│   ├── cleaning.py        # reglas de calidad (Silver)
│   ├── vmd.py             # VMD + optimización bayesiana
│   ├── kalman.py          # filtro de Kalman
│   ├── dataset.py         # ancla, ventanas y lotes en GPU
│   ├── mimo.py            # Min-Max y matrices para los modelos MIMO / grafo / TCN
│   ├── features.py        # RFR-SHAP
│   ├── soma.py            # SOMA (optimización de hiperparámetros)
│   ├── model.py           # celda NOA, Bi-NOA-LSTM, Seq2Seq, MIMO, ST-GRU, TCN, entrenamiento
│   └── metrics.py         # métricas y baselines
├── data/                  # raw, bronze, silver, gold (local, no versionado)
├── models/                # pesos (.pt, no versionados) y parámetros de los filtros (.json)
├── reports/               # métricas (.csv versionados) y figuras (.png, locales)
└── prompt.md              # especificación de la PoC
```

## Nuestro modelo

**Bi-LSTM con canal independiente** (`06_modelado`), elegida por tener el menor error y ser la más rápida (~2 min de entrenamiento, ~2 ms para pronosticar las 300 celdas). Sus diferencias con los modelos del Benchmarking:

1. **Pronóstico relativo al nivel reciente (ancla)** de la celda, en lugar del voltaje absoluto. Es lo que evita el sesgo de nivel que hunde a los modelos MIMO.
2. **Una celda por muestra con pesos compartidos** entre las 300 celdas (identidad por embedding), lo que multiplica los ejemplos de entrenamiento.
3. **Máscara de saltos de nivel:** al entrenar se ignoran los días posteriores a una intervención de mantenimiento, que no es predecible con los datos disponibles.
4. **Salida directa** de los 180 días en un solo pase, sin acumulación de error por recursión.

## Pendiente

- Notebook 08 (despliegue): función `predecir(fecha, celda)`, exportación de pronósticos y plan de monitoreo y reentrenamiento.
- Entrenamiento con varias semillas para confirmar las diferencias pequeñas entre modelos.
