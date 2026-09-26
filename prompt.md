# SYSTEM PROMPT: PoC MODELO PREDICTIVO VOLTCAST (JUPYTER / COLAB) — v3.2

> **Cambios v3.1** (tras el EDA del notebook 03): casos de celdas puenteadas corregidos, relación `[Volts]` / media corregida, nueva tabla de **saltos de nivel** en Silver (R9) y métricas separadas por orígenes que cruzan un salto en la evaluación.
>
> **Cambios v3.2** (notebooks 05–06): covariables estandarizadas sin días de paro y acotadas a ±`COV_CLIP`; en el entrenamiento se enmascaran los objetivos posteriores a un salto de nivel (`MASK_POST_JUMP`, saltos detectados solo con train), porque sin esa máscara la red sobreajusta a los saltos de train; tras el early stopping se reentrena con todo train (`REFIT_FULL_TRAIN`).

## OBJETIVO
Implementar, como una **serie de notebooks** organizada por **CRISP-DM** y con datos en **arquitectura Medallion (Bronze → Silver → Gold)**, la Prueba de Concepto (PoC) para pronosticar el **voltaje de cada celda individual** (181 celdas × 2 electrolizadores, EL A y EL B = 362 celdas) en una **fecha futura** dada, con la arquitectura **BO-VMD + Bi-NOA-LSTM (multi-horizonte directo)** y horizontes de **60 a 180 días**.

**Dataset:** `VoltajesDiariosDesde2020.xlsx`, con registros **diarios** del 2020-01-01 al 2026-08-27 (~2420 filas por hoja). `Dataset Sintetizado.xlsx` es un recorte del mismo origen (celdas 1–20, desde 2024) y se usa solo para pruebas rápidas.

La PoC se considera exitosa si:
- El modelo **supera al mejor baseline** (persistencia y tendencia lineal) en RMSE para los horizontes de 60 y 180 días.
- Se reporta y grafica la **curva de error vs. horizonte** (día 1 → 180), mostrando que no hay deriva por recursión (inferencia directa en un solo pase).
- Todas las métricas se calculan en **voltios (V) por celda**, contra la **señal real (sin filtrar)**.

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
│   ├── 05_preparacion_gold_bovmd.ipynb     # CRISP-DM 3  → Gold
│   ├── 06_modelado.ipynb                   # CRISP-DM 4
│   ├── 07_evaluacion.ipynb                 # CRISP-DM 5
│   └── 08_despliegue.ipynb                 # CRISP-DM 6
├── src/voltcast/                           # código compartido (sin duplicar entre notebooks)
│   ├── config.py      # rutas y parámetros
│   ├── io.py          # lectura Excel / escritura y lectura parquet por capa
│   ├── cleaning.py    # reglas Silver
│   ├── vmd.py         # VMD + optimización bayesiana
│   ├── dataset.py     # ventanas, ancla, Dataset/DataLoader
│   ├── model.py       # celda NOA, Bi-NOA-LSTM
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
TARGET_CELLS    = None      # None = todas (2 × 181); o lista, p. ej. ["A_c001", "B_c017"]
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
- Contexto: planta con electrolizadores EL A y EL B de 181 celdas en serie; por qué importa anticipar el voltaje de las celdas (consumo energético, degradación, mantenimiento).
- Objetivos de negocio vinculados a las historias de usuario US005–US008 (proyecciones, precisión, visualización, filtrado por fecha).
- Objetivo de minería de datos: pronóstico multi-horizonte (60–180 días) por celda.
- Criterios de éxito (los de la sección OBJETIVO), supuestos, restricciones y riesgos.
- Alcance: RFR-SHAP y SOMA de la Propuesta A quedan **fuera** de la PoC.

### 02 — Ingesta Bronze (CRISP-DM 2)
- Leer ambas hojas (`EL A 2020_2026`, `EL B 2020_2026`) con `openpyxl` en modo `read_only`.
- **Sin transformar valores**: solo se agregan las columnas `electrolizador` (A/B), `_source_file` e `_ingested_at`, y se convierten los nombres de columna a texto.
- Validaciones de esquema (US001/US002/US003): existe `Date`; se detectan las columnas de celdas numéricas (se **cuentan**, no se asume 181), `[Volts]` y `[kA]`. Si falla, error descriptivo.
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
- Renombrar a `A_V_total`, `A_kA`, `A_c001..A_c181` (ídem `B_`); descartar `Average`.
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
- Covariables por día: `kA` y `V_total / n_celdas_en_servicio` del electrolizador de la celda, estandarizadas con estadísticas de train.
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

### 07 — Evaluación (CRISP-DM 5)
- **Rolling-origin** sobre validación (orígenes cada `EVAL_STRIDE` días; el contexto de entrada puede venir de train).
- Inferencia: las 362 celdas como un batch → matriz `[180, 362]` en un único forward pass por origen. Reconstrucción `V̂ = ancla + σ_Δ · ŷ`.
- **Baselines:** persistencia (ancla) y tendencia lineal ajustada a la ventana de entrada.
- Métricas por celda y globales, en V, contra la señal **real sin filtrar** y solo en fechas con calidad `ok`/`interpolado`: `RMSE`, `MAE`, `MAPE`, `R²` (R² por celda, luego promediado).
- Tabla comparativa a 60 y 180 días: Bi-NOA-LSTM vs. Bi-LSTM estándar vs. baselines.
- Métricas reportadas también por separado para los pares (celda, origen) cuyo horizonte **cruza o no cruza** un salto de nivel (`saltos_nivel.parquet`).
- Gráficos:
  - Curva de **error vs. horizonte** (RMSE día 1..180) de modelos y baselines.
  - `Real vs. Predicho` a 60 y 180 días en celdas críticas (mayor voltaje y mayor error).
  - Mapa de calor del error por celda.
- Veredicto frente a los criterios de éxito, y decisión de continuar o iterar.
- Salida: `reports/metricas.csv`, figuras.

### 08 — Despliegue (CRISP-DM 6)
- Cargar el modelo elegido y los escaladores; pronosticar desde la última fecha disponible (VMD causal sobre los últimos `VMD_HISTORY` días).
- Función `predecir(fecha, celda)`, p. ej. `predecir("2026-12-01", "B_c017")`, que devuelve el voltaje en V.
  - Valida que `1 ≤ (fecha − origen) ≤ HORIZON_DAYS`, que la celda exista y que no esté fuera de servicio; si no, error descriptivo.
- Exportar el pronóstico completo en `data/gold/pronosticos.parquet` (`fecha_origen, fecha_objetivo, electrolizador, celda, voltaje_pred`), para US007/US008 (visualización y filtrado por fecha).
- Plan de monitoreo y reentrenamiento (US005-E02): frecuencia, métrica de alerta y versionado del modelo.

---

## RIESGOS CONOCIDOS
- **Datos de entrenamiento:** ~1935 días de train y ~484 de validación, con ~305 orígenes posibles para H=180, muy solapados. Hay pocas trayectorias independientes de 180 días.
- **Relación entre celdas:** el esquema de canal independiente no la modela explícitamente; se aproxima con covariables comunes del electrolizador (`kA`, `V_total`).
- **Parámetros de VMD comunes:** asumen un ruido parecido entre celdas; el notebook 05 lo verifica.
- **Costo de VMD causal en validación:** ~70 orígenes × 362 celdas ≈ 25 000 descomposiciones. Se limita con `EVAL_STRIDE` y `VMD_HISTORY`; conviene paralelizar (`joblib`).
- **Celdas puenteadas:** entre 2020 y ≈ oct-2024 hay 4–5 celdas puenteadas por electrolizador; esas muestras se excluyen.
- **Saltos de nivel por intervenciones:** ocurren cada ~6 meses, el mismo orden que el horizonte. Un pronóstico que cruza un salto tiene un error del orden de la caída, y no es predecible sin registro de mantenimientos.
