# SYSTEM PROMPT: PoC MODELO PREDICTIVO (JUPYTER / COLAB)

## OBJETIVO
Implementar en un Jupyter Notebook la Prueba de Concepto (PoC) para el pronóstico de voltajes en 183 celdas de alta tensión (más covariable \\(kA\\)) usando la arquitectura **BO-VMD + Bi-NOA-LSTM (MIMO)**, garantizando un \\(R^2 \ge 0.999\\), \\(\text{RMSE} < 0.70\text{ kW/V}\\) y cero acumulación de error en horizontes de 60 a 180 días.

---

## FLUJO DE EJECUCIÓN (5 BLOQUES)

### 1. Ingesta y Filtrado BO-VMD
- Carga el dataset horario continuo de 183 celdas + intensidad \\(kA\\).
- Aplica Descomposición Variacional de Modo Bayesiana (BO-VMD) para eliminar ruido de alta frecuencia y exporta `dataset_clean.parquet`.

### 2. Divisibilidad Temporal (80% Train / 20% Validation)
- Asigna el **20% final de las filas horarias** para validación fuera de la muestra (*Out-of-Sample*).
- **Cálculo y Log de Fechas:** Codifica el cálculo explícito de \\(N_{val} = N \times 0.20\\) e imprime en consola:
  `"El 20% de validación equivale a X filas horarias, abarcando desde [YYYY-MM-DD HH:MM] hasta [YYYY-MM-DD HH:MM]."`

### 3. Tensores MIMO (`PyTorch DataLoader`)
- **Entrada (\\(X\\)):** Ventana histórica \\(L = 720\\) horas (30 días) \\(\times\\) 184 variables \\(\to\\) Shape: `[Batch, 720, 184]`.
- **Salida (\\(Y\\)):** Matriz futura directa \\(H = 1440\text{ a }4320\\) horas (60 a 180 días) \\(\times\\) 183 celdas \\(\to\\) Shape: `[Batch, H, 183]`.

### 4. Modelo Bi-NOA-LSTM (`PyTorch`)
- Capa `nn.LSTM(bidirectional=True)`.
- **Mecanismo NOA:** Sin activación sigmoide/tanh en la capa densa de proyección para prevenir saturación de gradientes.
- **Inferencia MIMO Directa:** Salida de la matriz \\([Batch, H, 183]\\) en un único pase matricial (*single forward pass*).
- Entrena con loss MSE en GPU y guarda pesos en `modelo_poc.pt`.

### 5. Evaluación y Visualización
- Calcula métricas globales y por celda: \\(R^2\\), \\(\text{RMSE}\\), \\(\text{MAE}\\), \\(\text{MAPE}\\).
- Grafica curvas `Real vs. Predicho` a 60 y 180 días en celdas críticas para verificar la ausencia de *error drift*.

## IMPORTANTE

Revisa los otros md primero