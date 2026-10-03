# SYSTEM PROMPT: PoC COMPARATIVA DE MODELOS PREDICTIVOS (JUPYTER / COLAB)

## OBJETIVO
Implementar en Jupyter Notebooks una Prueba de Concepto (PoC) que **compare los 4 modelos mejor puntuados en el Benchmarking**, junto con **2 modelos de referencia (Persistencia y Tendencia Lineal)**, para el pronóstico de voltajes en 183 celdas de alta tensión (más la covariable \\(kA\\)). Todos se evalúan bajo las mismas condiciones con las métricas **MAE, RMSE, MSE, MAPE y \\(R^2\\)**.

El propósito es **validar empíricamente** el resultado del Benchmarking (que se basó en la literatura) con los datos reales de la planta, y confirmar o corregir la elección del modelo ganador. Los modelos de referencia permiten verificar que los modelos de Deep Learning aportan una mejora real frente a métodos simples. La PoC no debe asumir de antemano qué modelo es mejor: los resultados deben reportarse tal como salgan.

### Modelos a comparar (orden del Benchmarking)
| Puesto | Propuesta | Modelo | Puntaje Benchmarking |
| :--- | :--- | :--- | :--- |
| 1° | A | BO-VMD + SOMA-Bi-NOA-LSTM (MIMO) | 5.00 |
| 2° | C | ST-GRU (Spatial-Temporal GRU) | 4.00 |
| 3° | D | TCN Multi-Output | 3.90 |
| 4° | B | Seq2Seq BiLSTM con Atención + Filtro de Kalman | 3.80 |

La Propuesta E (Informer / PatchTST, 2.30) queda fuera de la PoC.

### Modelos de referencia (*baselines*)
| Modelo | Cómo pronostica | Entrenamiento |
| :--- | :--- | :--- |
| **Persistencia** | Repite el último voltaje observado de cada celda (hora \\(t\\)) para todas las horas del horizonte \\(H\\): \\(\hat{y}_{t+h} = y_t\\). | No requiere |
| **Tendencia Lineal** | Ajusta, para cada celda, una recta \\(y = a + b\,t\\) sobre la ventana histórica de \\(L = 720\\) horas y la extrapola a las \\(H\\) horas futuras. | Solo el ajuste de la recta en cada ventana |

---

## FLUJO DE EJECUCIÓN (5 BLOQUES)

### 1. Ingesta y Preprocesamiento
- Carga el dataset horario continuo de 183 celdas + intensidad \\(kA\\).
- Aplica a cada modelo **solo el filtrado que le corresponde** según `detalle-propuestas-modelos-ml.md` (BO-VMD para A, Filtro de Kalman para B; C y D usan la serie normalizada sin filtro externo). Los modelos de referencia usan la serie original sin filtrar.
- El filtrado se ajusta **solo con los datos de entrenamiento**, para no filtrar información del periodo de validación.

### 2. División Temporal (80% Train / 20% Validation)
- Usa **la misma división para los 6 modelos**: el **20% final de las filas horarias** se reserva para validación fuera de la muestra (*Out-of-Sample*).
- **Cálculo y Log de Fechas:** Codifica el cálculo explícito de \\(N_{val} = N \times 0.20\\) e imprime en consola:
  `"El 20% de validación equivale a X filas horarias, abarcando desde [YYYY-MM-DD HH:MM] hasta [YYYY-MM-DD HH:MM]."`
- Verifica que el periodo de validación alcance para al menos una ventana completa \\(L + H\\). Si no alcanza, infórmalo antes de continuar.

### 3. Tensores (`PyTorch DataLoader`)
- **Entrada (\\(X\\)):** Ventana histórica \\(L = 720\\) horas (30 días) \\(\times\\) 184 variables \\(\to\\) Shape: `[Batch, 720, 184]`.
- **Salida (\\(Y\\)):** Horizonte \\(H = 1440\text{ a }4320\\) horas (60 a 180 días) \\(\times\\) 183 celdas \\(\to\\) Shape: `[Batch, H, 183]`.
- Las mismas ventanas de entrada y salida se usan para entrenar y evaluar los 6 modelos.

### 4. Entrenamiento de los 4 Modelos y Cálculo de las Referencias (`PyTorch`)
- Implementa cada arquitectura según `detalle-propuestas-modelos-ml.md`, respetando su forma de generar el pronóstico (A y D: salida directa en un pase; B: decodificación paso a paso; C: proyección espaciotemporal).
- Entrena todos con loss MSE en GPU, con la misma semilla aleatoria y un presupuesto de entrenamiento comparable (épocas / *early stopping*).
- Guarda los pesos de cada modelo por separado: `modelo_A.pt`, `modelo_B.pt`, `modelo_C.pt`, `modelo_D.pt`.
- Calcula los pronósticos de Persistencia y Tendencia Lineal sobre las mismas ventanas de validación.
- Registra el tiempo de entrenamiento y el tiempo de inferencia de cada modelo.

### 5. Evaluación Comparativa y Visualización
- Evalúa las predicciones **desnormalizadas** contra los **voltajes reales sin filtrar** del periodo de validación, para que ningún modelo se compare contra una señal suavizada por su propio filtro.
- Calcula, para cada uno de los 6 modelos, las métricas globales, por celda y por horizonte (60 y 180 días):
  - \\(\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|\\)
  - \\(\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2\\)
  - \\(\text{RMSE} = \sqrt{\text{MSE}}\\)
  - \\(\text{MAPE} = \frac{100}{n}\sum \left|\frac{y - \hat{y}}{y}\right|\\)
  - \\(R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}\\)
- Genera una **tabla comparativa** (modelos \\(\times\\) métricas) y expórtala a `metricas_comparativas.csv`, indicando qué modelo obtiene el mejor valor en cada métrica.
- Indica, para cada modelo de Deep Learning, el porcentaje de mejora en MAE y RMSE frente a Persistencia y frente a Tendencia Lineal. Si algún modelo no supera a las referencias, repórtalo explícitamente.
- Grafica curvas `Real vs. Predicho` de los 6 modelos a 60 y 180 días en celdas críticas, para observar la acumulación de error (*error drift*).
- Concluye si el ranking empírico coincide con el del Benchmarking.

## IMPORTANTE

Revisa los otros md primero
