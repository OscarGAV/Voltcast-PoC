# SYSTEM PROMPT: PoC MODELO PREDICTIVO VOLTCAST (JUPYTER / COLAB) — v4

> **Cambios v3.1** (tras el EDA del notebook 03): casos de celdas puenteadas corregidos, relación `[Volts]` / media corregida, nueva tabla de **saltos de nivel** en Silver (R9) y métricas separadas por orígenes que cruzan un salto en la evaluación.
>
> **Cambios v4** (actualización de `docs/prompt_v1.md` y `docs/detalle-propuestas-modelos-ml.md`): la PoC pasa a ser **comparativa**. Se implementan los 4 modelos mejor puntuados del Benchmarking tal como los describe la literatura (A en dos variantes MIMO, B, C y D), se mantienen las dos variantes propias (Bi-NOA-LSTM y Bi-LSTM estándar con máscara de saltos) y se redefinen los baselines sobre la serie real. El dataset pasa a **150 celdas por electrolizador** (cantidad detectada automáticamente). Métricas: MAE, MSE, RMSE, MAPE y R².
>
> **Cambios v3.2** (notebooks 05–06): covariables estandarizadas sin días de paro y acotadas a ±`COV_CLIP`; en el entrenamiento se enmascaran los objetivos posteriores a un salto de nivel (`MASK_POST_JUMP`, saltos detectados solo con train), porque sin esa máscara la red sobreajusta a los saltos de train; tras el early stopping se reentrena con todo train (`REFIT_FULL_TRAIN`).

## OBJETIVO
Implementar, como una **serie de notebooks** organizada por **CRISP-DM** y con datos en **arquitectura Medallion (Bronze → Silver → Gold)**, una Prueba de Concepto (PoC) **comparativa** para pronosticar el **voltaje de cada celda individual** (2 electrolizadores, EL A y EL B; hoy **150 celdas** por electrolizador, cantidad detectada en la ingesta) con horizontes de **60 a 180 días**.

**Propósito:** demostrar con los datos reales de la planta que **nuestro modelo** supera a los 4 modelos mejor puntuados del Benchmarking (implementados tal como los describe la literatura) y a 2 modelos de referencia. Nuestro modelo se define a partir de las variantes propias (hoy Bi-NOA-LSTM y Bi-LSTM estándar, con las mejoras surgidas de los datos). Los resultados se reportan tal como salgan, aunque algún modelo del Benchmarking gane en alguna métrica.

**Modelos comparados (9):**

| Grupo | Modelo | Descripción |
|---|---|---|
| Benchmarking 1° (5.00) | **A1** BO-VMD + SOMA-Bi-NOA-LSTM, MIMO por electrolizador | `[B, 30, N + covariables]` → `[B, 180, N]`, un modelo por electrolizador |
| Benchmarking 1° (5.00) | **A2** BO-VMD + SOMA-Bi-NOA-LSTM, MIMO conjunto | `[B, 30, 2N + covariables]` → `[B, 180, 2N]`, un solo modelo |
| Benchmarking 2° (4.00) | **C** ST-GRU | GCN espacial sobre el grafo de celdas + GRU temporal |
| Benchmarking 3° (3.90) | **D** TCN Multi-Output | Convoluciones causales dilatadas + capa densa multi-salida, por electrolizador |
| Benchmarking 4° (3.80) | **B** Seq2Seq BiLSTM + atención + Kalman | Decodificación paso a paso; ensamble neuro-estocástico red + Kalman |
| Propio | **Bi-NOA-LSTM** | Canal independiente, ancla, covariables y máscara de saltos |
| Propio | **Bi-LSTM estándar** | Igual, con `nn.LSTM` |
| Referencia | **Persistencia** | `ŷ(t0 + h) = y(t0)`: último voltaje real observado |
| Referencia | **Tendencia lineal** | Recta ajustada a los 30 días de la serie real sin filtrar, extrapolada |

**Dataset:** `VoltajesDiariosDesde2020.xlsx`, con registros **diarios** del 2020-01-01 al 2026-08-27 (~2420 filas por hoja), 150 celdas por electrolizador. `[Volts]` sigue sumando las 181 celdas físicas (`N_CELDAS_FISICAS`). `Dataset Sintetizado.xlsx` es un recorte del mismo origen y se usa solo para pruebas rápidas.

**Adaptaciones al dataset real** (el Benchmarking supone datos horarios y 183 celdas): `L = 30 días` (≈ 720 h), `H = 60–180 días` (≈ 1440–4320 h), mensajes en "filas diarias", 150 celdas por electrolizador.

**Evaluación:**
- Todos los modelos con el mismo split 80/20, las mismas fechas de origen de validación, la misma semilla (`SEED`) y un presupuesto de entrenamiento comparable (early stopping).
- Métricas **MAE, MSE, RMSE, MAPE y R²**, globales, por celda y por horizonte (60 y 180 días), en **voltios** y contra la **señal real sin filtrar**, desnormalizando cada modelo.
- `metricas_comparativas.csv` con el mejor modelo por métrica; % de mejora en MAE y RMSE de cada red frente a Persistencia y a Tendencia lineal (reportando explícitamente si alguna no los supera); tiempos de entrenamiento e inferencia; `Real vs. Predicho` de todos los modelos en celdas críticas; curva de error vs. horizonte (*error drift*); conclusión sobre si el ranking empírico coincide con el del Benchmarking.

> Nota: la inferencia directa elimina la acumulación de error *recursiva*, pero la incertidumbre crece naturalmente con el horizonte. Eso se mide; no se asume que sea nulo.

---

## ENTORNO
- **Python 3.13** en un entorno virtual **`.venv`** en la raíz del proyecto:
  `py -3.13 -m venv .venv` → `.venv\Scripts\python -m pip install -r requirements.txt`.
- `requirements.txt` con versiones fijadas y compatibles con 3.13: `pandas`, `numpy`, `pyarrow`, `openpyxl`, `matplotlib`, `seaborn`, `scikit-learn`, `optuna`, `vmdpy`, `joblib`, `torch`, `ipykernel`.
- Registrar el kernel (`python -m ipykernel install --user --name voltcast-venv`) y ejecutar todos los notebooks con él.
- `.venv/` y `data/` fuera del control de versiones (`.gitignore`).
- **Git:** el proyecto vive en un repositorio git. Cada avance importante (cada notebook o capa Medallion terminada, cambios de especificación) se commitea **antes de empezar el siguiente**, con un mensaje descriptivo. Los notebooks se commitean sin salidas pesadas (`nbstripout` o limpiar outputs antes del commit).

---

## ESTRUCTURA DEL PROYECTO

```
Poc-TP1/                                    # C:\Proyectos\Poc-TP1 (fuera de OneDrive)
├── docs/                                   # propuestas de modelos, historias de usuario, prompt v1
├── notebooks/
│   ├── 01_comprension_negocio.ipynb        # CRISP-DM 1
│   ├── 02_ingesta_bronze.ipynb             # CRISP-DM 2  → Bronze
│   ├── 03_comprension_datos.ipynb          # CRISP-DM 2  (EDA sobre Bronze)
│   ├── 04_preparacion_silver.ipynb         # CRISP-DM 3  → Silver
│   ├── 05_preparacion_gold_bovmd.ipynb     # CRISP-DM 3  → Gold (BO-VMD)
│   ├── 05b_preparacion_gold_kalman.ipynb   # CRISP-DM 3  → Gold (Kalman, Propuesta B)
│   ├── 06_modelado.ipynb                   # CRISP-DM 4  variantes propias (Bi-NOA-LSTM, Bi-LSTM)
│   ├── 06b_modelado_seq2seq_kalman.ipynb   # CRISP-DM 4  Propuesta B
│   ├── 06c_modelado_propuesta_a.ipynb      # CRISP-DM 4  Propuesta A (A1, A2): RFR-SHAP + SOMA + MIMO
│   ├── 06d_modelado_stgru_tcn.ipynb        # CRISP-DM 4  Propuestas C y D
│   ├── 07_evaluacion.ipynb                 # CRISP-DM 5  comparativa de los 9 modelos
│   └── 08_despliegue.ipynb                 # CRISP-DM 6
├── src/voltcast/                           # código compartido (sin duplicar entre notebooks)
│   ├── config.py      # rutas y parámetros
│   ├── io.py          # lectura Excel / escritura y lectura parquet por capa
│   ├── cleaning.py    # reglas Silver
│   ├── vmd.py         # VMD + optimización bayesiana
│   ├── dataset.py     # ventanas, ancla, Dataset/DataLoader
│   ├── model.py       # celda NOA, Bi-NOA-LSTM, Seq2Seq, MIMO, ST-GRU, TCN
│   ├── kalman.py      # filtro de Kalman (Propuesta B)
│   ├── mimo.py        # Min-Max y matrices para los modelos MIMO / grafo / TCN
│   ├── soma.py        # SOMA (optimización de hiperparámetros, Propuesta A)
│   ├── features.py    # RFR-SHAP (selección de covariables, Propuesta A)
│   └── metrics.py     # métricas y baselines
├── data/
│   ├── raw/           # Excel originales (solo lectura)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── models/            # pesos y artefactos del modelo
└── reports/           # figuras y tablas de resultados
```

### Reglas comunes a todos los notebooks
- **Encabezado estándar:** fase CRISP-DM, objetivo, entradas (capa/archivo), salidas (capa/archivo).
- Cada notebook **lee solo de la capa anterior** y **escribe solo en su capa**. Nunca se lee el Excel después de Bronze.
- Cada notebook se puede ejecutar de forma independiente, a partir de los archivos de la capa anterior.
- Toda la lógica reutilizable vive en `src/voltcast/`; los notebooks orquestan, muestran y documentan.
- Compatibles con Colab: `config.py` detecta si corre en Colab (monta Drive y ajusta `BASE_DIR`) o en local.
- Formato de las capas: Parquet (`pyarrow`). Cada escritura va acompañada de un `_metadata.json` con fecha de ejecución, filas, columnas, archivo de origen y hash del origen (trazabilidad).

### `config.py` (parámetros centrales)
```python
RAW_FILE        = "VoltajesDiariosDesde2020.xlsx"   # o "Dataset Sintetizado.xlsx" para pruebas rápidas
LOOKBACK_DAYS   = 30        # L
HORIZON_DAYS    = 180       # H máx.; se evalúan cortes a 60 y 180
EVAL_HORIZONS   = [60, 180]
VAL_FRACTION    = 0.20
TARGET_CELLS    = None      # None = todas las detectadas; o lista, p. ej. ["A_c001", "B_c017"]
N_CELDAS_FISICAS = 181      # celdas físicas por electrolizador ([Volts] las suma todas)
ANCHOR_DAYS     = 7         # días promediados para el ancla del nivel
KA_SHUTDOWN     = 10.0      # kA por debajo = día de paro
MAX_INTERP_DAYS = 3         # huecos ≤ 3 días se interpolan
OOS_MIN_DAYS    = 7         # duración mínima de un tramo fuera_servicio
OOS_MERGE_DAYS  = 5         # interrupciones ≤ 5 días no cortan un tramo fuera_servicio
STEP_THRESHOLD_V = 0.10     # cambio de nivel (V) para registrar un salto
BO_SAMPLE_CELLS = 20        # celdas para BO-VMD (10 por electrolizador, muestreo estratificado)
BO_TRIALS       = 30        # evaluaciones de la optimización bayesiana por celda de la muestra
EVAL_STRIDE     = 7         # días entre orígenes de validación (rolling-origin)
VMD_HISTORY     = 365       # días de historia usados para VMD causal en validación/inferencia
USE_NOA         = True      # True = Bi-NOA-LSTM; False = nn.LSTM estándar (ablación)
MASK_POST_JUMP  = True      # enmascarar en el loss los objetivos posteriores a un salto de nivel (histórico de train)
ES_BLOCK_DAYS   = 270       # último bloque de train para early stopping
REFIT_FULL_TRAIN = True     # reentrenar con todo train durante las épocas elegidas
COV_CLIP        = 5.0       # |z| máximo de las covariables
SEED            = 42
```

---

## NOTEBOOKS

### 01 — Comprensión del negocio (CRISP-DM 1)
Notebook mayoritariamente en Markdown:
- Contexto: planta con electrolizadores EL A y EL B de 181 celdas en serie (el dataset incluye 150 por electrolizador); por qué importa anticipar el voltaje de las celdas (consumo energético, degradación, mantenimiento).
- Objetivos de negocio vinculados a las historias de usuario US005–US008 (proyecciones, precisión, visualización, filtrado por fecha).
- Objetivo de minería de datos: pronóstico multi-horizonte (60–180 días) por celda.
- Criterios de éxito (los de la sección OBJETIVO), supuestos, restricciones y riesgos.
- Alcance: comparación de los 4 modelos del Benchmarking (A1, A2, B, C, D) con las variantes propias y los baselines. La Propuesta E queda fuera.

### 02 — Ingesta Bronze (CRISP-DM 2)
- Leer ambas hojas (`EL A 2020_2026`, `EL B 2020_2026`) con `openpyxl` en modo `read_only`.
- **Sin transformar valores**: solo se agregan las columnas `electrolizador` (A/B), `_source_file` e `_ingested_at`, y se convierten los nombres de columna a texto.
- Validaciones de esquema (US001/US002/US003): existe `Date`; se detectan las columnas de celdas numéricas (se **cuentan**: hoy 150, el sistema se adapta si aumentan o disminuyen), `[Volts]` y `[kA]`. Si falla, error descriptivo. Un cambio en la cantidad o la numeración de las celdas obliga a reentrenar.
- Salida: `data/bronze/voltajes_EL_A.parquet`, `data/bronze/voltajes_EL_B.parquet` + metadata.

### 03 — Comprensión de los datos (CRISP-DM 2)
EDA sobre Bronze, con figuras guardadas en `reports/eda/`:
- Rango de fechas, frecuencia, hora de registro (la mayoría a las 00:00; ~30 % entre las 07:00 y las 11:00), fechas duplicadas.
- Faltantes por fecha y por celda (mapa de calor celdas × tiempo).
- Días de paro (kA bajo) y su efecto sobre los voltajes.
- Celdas fuera de servicio o puenteadas: V < 0.5 o negativo con kA normal. Casos detectados: celdas 61, 62, 123 y 124 de ambos electrolizadores desde 2020-01 hasta ≈ oct-2024 (EL A 123/124 hasta ≈ abr-2025); hasta may-2022 con V < 0.5 y luego mayormente sin dato. Celda 46 del EL B (jul-2021 → ene-2022) y tramos cortos en las celdas 18, 60, 80, 122 y 132 del EL B.
- Tendencia de envejecimiento (`[Volts]` y voltaje medio de las celdas a lo largo del tiempo), estacionalidad y dispersión entre celdas.
- Relación `[Volts]` / media de las celdas en servicio (≈ 180 con 177 celdas, ≈ 186 con 181: `[Volts]` ≈ Σ celdas + 9–15 V) y correlación de las celdas con kA.
- Saltos de nivel por celda (intervenciones): caídas bruscas de 0.1–0.7 V en lotes de celdas, ~12–14 fechas por electrolizador (cada ~6 meses).
- Conclusión: lista de reglas de calidad que se aplicarán en Silver.

### 04 — Preparación Silver (CRISP-DM 3)
- Renombrar a `A_V_total`, `A_kA`, `A_c001..A_cNNN` (ídem `B_`; NNN = celdas detectadas); descartar `Average`.
- `Date` → **fecha calendario**; ordenar; en fechas duplicadas, quedarse con el último registro del día; reindexar a frecuencia diaria continua.
- Interpolar linealmente los huecos de hasta `MAX_INTERP_DAYS`; los huecos más largos quedan como `NaN`.
- **Máscara de calidad** por celda y fecha, con códigos: `ok`, `interpolado`, `hueco`, `paro`, `fuera_servicio`, `atipico`.
  - `paro`: `kA < KA_SHUTDOWN`.
  - `fuera_servicio`: detección automática de tramos con V < 0.5 (incluye negativos) con kA normal, o sin datos durante un período largo mientras el resto de las celdas sí tiene dato. Tramos de al menos `OOS_MIN_DAYS` días, uniendo interrupciones de hasta `OOS_MERGE_DAYS` días; los días de paro o sin registro no cortan un tramo.
  - `atipico`: V > 5 V o V < 0 aislados → `NaN`.
- **Split temporal 80/20** (se decide aquí y se guarda para todas las capas siguientes):
  - Calcular `N_val = round(N * VAL_FRACTION)` e imprimir:
    `"El 20% de validación equivale a X filas diarias, abarcando desde [YYYY-MM-DD] hasta [YYYY-MM-DD]."`
  - Guardar `data/silver/split.json` (fecha de corte).
- Formato largo recomendado: `fecha, electrolizador, celda, voltaje, kA, V_total, calidad`.
- **Saltos de nivel (R9):** tabla de eventos por celda (`fecha, electrolizador, celda, delta_V`), con cambios de más de `STEP_THRESHOLD_V` entre la mediana de los 7 días previos y la de los 7 siguientes, sobre datos `ok`/`interpolado`. **No** se corrigen los valores. Solo se usa para diagnóstico en la evaluación, **nunca como entrada del modelo** (los saltos posteriores al origen son información del futuro).
- Salida: `data/silver/voltajes_silver.parquet`, `data/silver/split.json`, `data/silver/saltos_nivel.parquet`, resumen de calidad + metadata.

### 05 — Preparación Gold: BO-VMD y features (CRISP-DM 3)
**Todo ajuste usa solo train.**

**Optimización bayesiana sobre una muestra representativa:**
- Elegir `BO_SAMPLE_CELLS` = 20 celdas (10 por electrolizador), estratificadas por posición en el stack (inicio, medio y final) y por nivel de voltaje. Excluir celdas con tramos `fuera_servicio`.
- Para cada celda de la muestra, optimización bayesiana (`optuna`, TPE, `BO_TRIALS` pruebas) de `K` (p. ej. 3–10) y `alpha` (p. ej. 500–5000, escala logarítmica) de VMD (`vmdpy`), minimizando la **entropía de envolvente** de los modos.
- **Verificación de homogeneidad:** tabla y gráfico de los `K` y `alpha` óptimos de la muestra.
  - Si son parecidos, se usa un par común (la mediana).
  - Si difieren mucho entre electrolizadores, se usa un par por electrolizador. Documentar la decisión.
- Guardar `models/vmd_params.json`.

**Filtrado:**
- Reconstruir la señal filtrada descartando el modo de más alta frecuencia (documentar cuántos modos se descartan).
- **Train:** VMD sobre el segmento de train de cada celda.
- **Validación (causal):** para cada origen `t0` (cada `EVAL_STRIDE` días), VMD solo sobre los últimos `VMD_HISTORY` días disponibles hasta `t0`. Nunca se descompone la serie completa.

**Features y objetivo:**
- `ancla = media de los últimos ANCHOR_DAYS días` de la señal filtrada de la celda.
- Objetivo: `y_h = (V_real(t0 + h) − ancla) / σ_Δ`, con `σ_Δ` = desviación estándar global de esas diferencias en train.
- Covariables por día: `kA` y `V_total / n_celdas_en_servicio` del electrolizador de la celda (las celdas físicas fuera del Excel cuentan como en servicio, `N_CELDAS_FISICAS`), estandarizadas con estadísticas de train.
- Salida: `data/gold/series_filtradas.parquet`, `data/gold/escaladores.json` (`σ_Δ`, medias y desviaciones), `data/gold/origenes_validacion.parquet` + metadata.
  Las ventanas se construyen en memoria en el notebook 06 a partir de estos archivos, porque materializarlas ocuparía demasiado espacio.

### 06 — Modelado (CRISP-DM 4)
**Esquema de canal independiente con pesos compartidos:**
- Una muestra = (celda, origen `t0`).
- **Entrada:** `[B, 30, F]`, con voltaje filtrado relativo al ancla (÷ `σ_Δ`), `kA`, `V_total` normalizado y canal-máscara.
- Identidad de la celda: `nn.Embedding(n_celdas, 8)` + indicador de electrolizador.
- **Salida:** `[B, 180]` (horizonte completo en un solo pase).
- Se excluyen las muestras cuya ventana de entrada cae en un tramo `fuera_servicio`; los objetivos enmascarados se excluyen del loss (**MSE enmascarado**).
- Con `MASK_POST_JUMP`, también se enmascaran los objetivos a partir del próximo salto de nivel de la celda (saltos detectados con la regla R9 **solo sobre train**). El modelo aprende la trayectoria sin intervenciones; la evaluación usa todos los objetivos válidos.

**Celda Bi-NOA-LSTM (propia):**
- LSTM estándar, salvo la salida del estado oculto: `h_t = o_t ⊙ c_t` (sin `tanh`).
- Bidireccional; implementada con `torch.jit.script`.
- Estabilidad: *gradient clipping* (norma 1.0) y dropout.

**Cabezal y entrenamiento:**
- Cabezal: `concat(h_final, embedding_celda, flag_EL) → Linear → Linear(→ 180)`, sin activación de salida.
- Entrenamiento: AdamW, *early stopping* sobre el último bloque de train (**no** sobre validación), GPU si está disponible, orígenes submuestreados (stride 3–7 días, fase aleatoria por época) para acortar las épocas. En el bloque de early stopping, los objetivos del train interno se cortan antes del bloque. Con la época elegida se reentrena con todo train (`REFIT_FULL_TRAIN`).
- **Ablación:** entrenar también con `USE_NOA=False` (`nn.LSTM(bidirectional=True)`).
- Salida: `models/bi_noa_lstm.pt`, `models/bi_lstm_baseline.pt` (pesos, config, escaladores, versión) y curvas de entrenamiento en `reports/`.

### 05b — Preparación Gold: filtro de Kalman (Propuesta B)
- Modelo de nivel y tendencia local; varianzas por máxima verosimilitud **solo con train** (reiniciando el filtro en saltos y arranques solo para estimar).
- Filtro causal sobre la serie completa (no requiere recalcular por origen). Salida: `data/gold/series_kalman.parquet` (nivel y pendiente), `data/gold/escaladores_kalman.json`, `models/kalman_params.json`.

### 06b — Propuesta B: Seq2Seq BiLSTM + atención + Kalman
- Encoder BiLSTM, decoder LSTM con atención multi-cabeza y **decodificación paso a paso** (Gou et al., 2024), sobre la señal de Kalman. Una semilla (`SEED`), sin máscara de saltos.
- **Ensamble neuro-estocástico** (Khan et al., 2026): `V̂ = α · V̂_red + (1 − α) · V̂_Kalman`, con `V̂_Kalman = nivel + pendiente · h` y `α ∈ [0,3; 0,7]` elegido en el bloque final de train.
- Salida: `models/modelo_B.pt`.

### 06c — Propuesta A tal cual el Benchmarking (A1 y A2)
- **RFR-SHAP:** Random Forest + SHAP sobre covariables candidatas (kA, voltaje medio por celda, calendario) para seleccionar las que entran al modelo (umbral de importancia documentado).
- **Min-Max** por columna ajustado con train (entradas BO-VMD y objetivos reales); desnormalización inversa para evaluar.
- **SOMA** (All-To-One) para tasa de aprendizaje, capas y unidades ocultas, evaluando en el bloque final de train.
- **Bi-NOA-LSTM MIMO** con proyección densa final sin activación y salida directa. A1: un modelo por electrolizador; A2: un modelo conjunto. Sin máscara de saltos; MSE enmascarado solo en objetivos no válidos.
- Salida: `models/modelo_A1.pt`, `models/modelo_A2.pt`, resultados de RFR-SHAP y SOMA en `reports/modelado/`.

### 06d — Propuestas C (ST-GRU) y D (TCN Multi-Output)
- Serie real normalizada (Min-Max, train) sin filtro externo; huecos de entrada rellenados con el último valor anterior (causal).
- **C:** grafo por electrolizador con las 2 vecinas en serie + las 4 celdas de mayor correlación en train; GCN espacial por día + GRU temporal por celda + cabezal lineal.
- **D:** convolución de suavizado + bloques residuales causales dilatados (1, 2, 4, 8, 16) + capa densa multi-salida, un modelo por electrolizador.
- Sin máscara de saltos. Salida: `models/modelo_C.pt`, `models/modelo_D.pt`.

### 07 — Evaluación (CRISP-DM 5)
- **Rolling-origin** sobre validación (orígenes cada `EVAL_STRIDE` días; el contexto de entrada puede venir de train).
- Inferencia: las 362 celdas como un batch → matriz `[180, 362]` en un único forward pass por origen. Reconstrucción `V̂ = ancla + σ_Δ · ŷ`.
- **Baselines:** persistencia (`ŷ = y(t0)`, último voltaje real observado) y tendencia lineal ajustada a los 30 días de la serie real sin filtrar.
- Métricas por celda y globales, en V, contra la señal **real sin filtrar** y solo en fechas con calidad `ok`/`interpolado`: `MAE`, `MSE`, `RMSE`, `MAPE`, `R²` (global, por celda y **sobre el cambio de voltaje** respecto del valor real en `t0`, que mide qué parte de la evolución anticipa el modelo sin el efecto del nivel de cada celda).
- Tabla comparativa de los 9 modelos (global 1–180, 60 y 180 días) → `reports/metricas_comparativas.csv`, con el mejor modelo por métrica, % de mejora frente a los baselines, tiempos de entrenamiento e inferencia y ranking empírico vs. Benchmarking.
- Métricas reportadas también por separado para los pares (celda, origen) cuyo horizonte **cruza o no cruza** un salto de nivel (`saltos_nivel.parquet`).
- Gráficos:
  - Curva de **error vs. horizonte** (RMSE día 1..180) de modelos y baselines.
  - `Real vs. Predicho` a 60 y 180 días en celdas críticas (mayor voltaje y mayor error).
  - Mapa de calor del error por celda.
- Veredicto frente a los criterios de éxito, y decisión de continuar o iterar.
- Salida: `reports/metricas.csv`, figuras.

### 08 — Despliegue (CRISP-DM 6)
- **Modelo elegido:** Bi-LSTM propia (`models/bi_lstm_baseline.pt`), mejor en todas las métricas sobre el horizonte completo en el notebook 07.
- Cargar el modelo elegido y los escaladores; pronosticar desde la última fecha disponible (VMD causal sobre los últimos `VMD_HISTORY` días).
- Función `predecir(fecha, celda)`, p. ej. `predecir("2026-12-01", "B_c017")`, que devuelve el voltaje en V.
  - Valida que `1 ≤ (fecha − origen) ≤ HORIZON_DAYS`, que la celda exista y que no esté fuera de servicio; si no, error descriptivo.
- Exportar el pronóstico completo en `data/gold/pronosticos.parquet` (`fecha_origen, fecha_objetivo, electrolizador, celda, voltaje_pred`), para US007/US008 (visualización y filtrado por fecha).
- Plan de monitoreo y reentrenamiento (US005-E02): frecuencia, métrica de alerta y versionado del modelo.

---

## PENDIENTES DE REVISIÓN
- **Split 90/10:** evaluar `VAL_FRACTION = 0.10` (train hasta 2025-12-25, validación de 243 días). Se mantiene 80/20 porque con 90/10 la validación baja a ~35 orígenes y ~10 con horizonte completo de 180 días (menos evidencia estadística), dominada por la intervención de jul-2026. Si se aprueba, repetir todo desde el notebook 04, incluido SOMA (~4 h).
- **Despliegue:** usar el modelo evaluado (entrenado con el 80 %); el reentrenamiento con todos los datos queda para el plan de reentrenamiento del notebook 08.

## RIESGOS CONOCIDOS
- **Datos de entrenamiento:** ~1935 días de train y ~484 de validación, con ~305 orígenes posibles para H=180, muy solapados. Hay pocas trayectorias independientes de 180 días.
- **Relación entre celdas:** el esquema de canal independiente no la modela explícitamente; se aproxima con covariables comunes del electrolizador (`kA`, `V_total`).
- **Parámetros de VMD comunes:** asumen un ruido parecido entre celdas; el notebook 05 lo verifica.
- **Costo de VMD causal en validación:** ~70 orígenes × 362 celdas ≈ 25 000 descomposiciones. Se limita con `EVAL_STRIDE` y `VMD_HISTORY`; conviene paralelizar (`joblib`).
- **Celdas puenteadas:** entre 2020 y ≈ oct-2024 hay 4–5 celdas puenteadas por electrolizador; esas muestras se excluyen.
- **Saltos de nivel por intervenciones:** ocurren cada ~6 meses, el mismo orden que el horizonte. Un pronóstico que cruza un salto tiene un error del orden de la caída, y no es predecible sin registro de mantenimientos.
