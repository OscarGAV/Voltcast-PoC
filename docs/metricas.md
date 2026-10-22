
## Criterios de evaluación del modelo

Todas las métricas se calculan **en voltios por celda**, desnormalizando cada modelo y comparando contra el **voltaje real sin filtrar** del período de validación (fuera de muestra), solo en fechas con voltaje **medido** (calidad `ok`): los valores interpolados se usan como entrada del modelo, pero no como objetivo ni en las métricas (`docs/hiperparametros.md`, sección 6.2). Se reportan sobre el horizonte completo (1–180 días) y en los cortes de **60 y 180 días**.

### 1. Requisito de la documentación del proyecto (OB2 – Diseño de Arquitectura, Tabla 10 y RNF-06)

Son los criterios de evaluación del modelo definidos en la documentación del proyecto. Se miden en fechas que el modelo no usó al entrenarse.

| Métrica | Justificación | Criterio esperado |
| --- | --- | --- |
| MAE (Mean Absolute Error) | Promedio de los errores absolutos entre predicciones y valores reales. Interpretación lineal y directa de la magnitud del error en voltios. | MAE < 5 % respecto al voltaje real (RNF-06); con la media de las celdas en validación, ≈ 3,1 V → MAE < 0,155 V. |
| RMSE (Root Mean Squared Error) | Raíz de la media de los errores al cuadrado. Penaliza las desviaciones grandes y permite identificar picos de error o fallas graves en la estimación. | RMSE < 0,15 V, para indicar que no hay errores grandes aislados. |
| MSE (Mean Squared Error) | Media de los errores al cuadrado. Es la función de pérdida (*loss*) del entrenamiento. | En entrenamiento: el error deja de bajar en iteraciones consecutivas (early stopping). En validación: MSE < 0,0225 V² (equivale a RMSE < 0,15 V). |
| MAPE (Mean Absolute Percentage Error) | Porcentaje de error absoluto relativo al valor real. Evalúa la precisión de forma independiente del nivel de voltaje de cada celda. | MAPE < 5 % en las fechas que el modelo no usó al entrenarse. |
| R² (coeficiente de determinación) | Indica si el modelo es más preciso que una predicción basada en el promedio. | R² > 0 en las fechas que el modelo no usó al entrenarse (R² global; el R² por celda se reporta sin umbral). |

> **Nota:** la documentación indica TensorFlow como framework de entrenamiento; la PoC se implementó en PyTorch. Las métricas no dependen del framework.

> **Limitación:** con estos umbrales también cumple el modelo más simple (persistencia: repetir el último voltaje observado). Verifican que el error es aceptable, pero **no** demuestran que un modelo es mejor que otro. Para eso está la comparación con el Benchmarking y los criterios comparativos.

### Requisito de finalización de la PoC

La PoC se considera finalizada cuando un modelo propio:
1. cumple todos los criterios de la sección 1 en el horizonte completo (1–180 días) y a 60 y 180 días; y
2. supera a los modelos del Benchmarking (A1, A2, B, C y D) en MAE, MSE, RMSE, MAPE y R², evaluados en las mismas condiciones (mismo split 80/20, mismos objetivos, voltaje real sin filtrar).

### 2. Criterios comparativos (propios de la PoC, complementarios)

No forman parte de la documentación del proyecto ni del requisito de finalización. Son más exigentes y se reportan como evidencia adicional de que nuestro modelo es mejor, con sus límites explicados.

| Criterio | Qué exige | Por qué |
| --- | --- | --- |
| **C1 · Supera a los baselines** | MAE y RMSE menores que los de la **persistencia**, la **tendencia lineal** y **ETS** (suavizado exponencial de Holt, el método del pronóstico integrado de Power BI). | Un modelo que no supera a un método sin entrenamiento, o a la herramienta que la planta ya podría usar, no aporta valor. |
| **C2 · Supera al Benchmarking con significancia** | RMSE menor que el del mejor modelo del Benchmarking, con intervalo de confianza del 95 % (bootstrap por fechas de origen) que no incluye el cero. | Evita declarar ganador por diferencias que podrían deberse al azar. |
| **C3 · Error acotado por celda** | RMSE < 0,15 V en **al menos el 95 % de las celdas**. | El promedio global puede ocultar celdas con error alto, que son justamente las críticas. |
| **C4 · Anticipa la evolución** | **R² sobre el cambio de voltaje** > 0: compara el cambio real `y(t0 + h) − y(t0)` con el predicho. | El R² global está inflado por las diferencias de nivel entre celdas (hasta la persistencia obtiene ≈ 0,73); este R² mide si el modelo anticipa cómo cambiará cada celda. |

Las métricas se reportan además por separado para los pronósticos que **cruzan una intervención de mantenimiento** (salto de nivel) y los que no, porque las intervenciones no son predecibles con los datos disponibles.

### 3. Dónde se evalúan

- Notebook `notebooks/07_evaluacion.ipynb`, sección 10 (*Cumplimiento de los criterios de `docs/metricas.md`*).
- Resultados por modelo y horizonte: `reports/evaluacion/cumplimiento_metricas.csv`; métricas completas: `reports/metricas_comparativas.csv`.
