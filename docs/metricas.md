
## Criterios de evaluación del modelo

Todas las métricas se calculan **en voltios por celda**, desnormalizando cada modelo y comparando contra el **voltaje real sin filtrar** del período de validación (fuera de muestra), solo en fechas con dato válido. Se reportan sobre el horizonte completo (1–180 días) y en los cortes de **60 y 180 días**.

### 1. Requisito mínimo (RNF06)

| Métrica | Justificación | Criterio esperado |
| --- | --- | --- |
| MAE (Mean Absolute Error) | Promedio de los errores absolutos entre predicciones y valores reales. Interpretación lineal y directa de la magnitud del error en voltios. | MAE < 5 % del voltaje nominal **de una celda** (≈ 3,1 V → MAE < 0,155 V). |
| RMSE (Root Mean Squared Error) | Raíz de la media de los errores al cuadrado. Penaliza las desviaciones grandes y permite identificar picos de error o fallas graves en la estimación. | RMSE < 0,15 V. |
| MSE (Mean Squared Error) | Media de los errores al cuadrado. Se usa como función de pérdida (*loss*) al entrenar las redes en PyTorch y cuantifica la varianza del error. | MSE < 0,0225 V² (equivale a RMSE < 0,15 V). |
| MAPE (Mean Absolute Percentage Error) | Porcentaje de error absoluto relativo al valor real. Evalúa la precisión de forma independiente del nivel de voltaje de cada celda. | MAPE < 5 % global en el conjunto de validación. |
| R² (coeficiente de determinación) | Proporción de la variación de los voltajes reales que explica el modelo. | Se reporta (global y por celda); sin umbral. |

> **Limitación del requisito mínimo:** con estos umbrales, el modelo más simple (persistencia: repetir el último voltaje observado) y casi todos los modelos del Benchmarking también cumplen. Sirven para verificar que el error es aceptable, pero **no** para demostrar que un modelo es mejor que otro. Para eso están los criterios comparativos.

### 2. Criterios comparativos (para demostrar que nuestro modelo es mejor)

| Criterio | Qué exige | Por qué |
| --- | --- | --- |
| **C1 · Supera a los baselines** | MAE y RMSE menores que los de la **persistencia** y la **tendencia lineal**. | Un modelo que no supera a un método sin entrenamiento no aporta valor. |
| **C2 · Supera al Benchmarking con significancia** | RMSE menor que el del mejor modelo del Benchmarking, con intervalo de confianza del 95 % (bootstrap por fechas de origen) que no incluye el cero. | Evita declarar ganador por diferencias que podrían deberse al azar. |
| **C3 · Error acotado por celda** | RMSE < 0,15 V en **al menos el 95 % de las celdas**. | El promedio global puede ocultar celdas con error alto, que son justamente las críticas. |
| **C4 · Anticipa la evolución** | **R² sobre el cambio de voltaje** > 0: compara el cambio real `y(t0 + h) − y(t0)` con el predicho. | El R² global está inflado por las diferencias de nivel entre celdas (hasta la persistencia obtiene ≈ 0,73); este R² mide si el modelo anticipa cómo cambiará cada celda. |

Las métricas se reportan además por separado para los pronósticos que **cruzan una intervención de mantenimiento** (salto de nivel) y los que no, porque las intervenciones no son predecibles con los datos disponibles.

### 3. Dónde se evalúan

- Notebook `notebooks/07_evaluacion.ipynb`, sección 10 (*Cumplimiento de los criterios de `docs/metricas.md`*).
- Resultados por modelo y horizonte: `reports/evaluacion/cumplimiento_metricas.csv`; métricas completas: `reports/metricas_comparativas.csv`.
