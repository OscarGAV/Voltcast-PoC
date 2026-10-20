# Intentos de mejora del modelo propio (rama `feature/test`)

Objetivo: cerrar los criterios de `docs/metricas.md` que quedan abiertos para la Bi-LSTM (C1 y C2 a 60 días, C3 a 180 días, C4).

**Protocolo.** Todo cambio se evalúa primero con validación cruzada temporal dentro de train (bloques F1 2022-07 → 2023-06, F2 2023-07 → 2024-07 y F3 = últimos 270 días de train; ver notebook `06e`). En cada bloque, el modelo se entrena solo con orígenes y objetivos anteriores al bloque, con VMD causal y contra el voltaje real. La validación del notebook 07 se usa una sola vez por candidato.

## Diagnóstico

- En validación, la Bi-LSTM ya es la mejor en los objetivos que **no cruzan** una intervención de mantenimiento (RMSE a 60 días 0,0331 V frente a 0,0357 V de la persistencia).
- Todo el déficit viene del **2,7 % de objetivos que cruzan un salto de nivel**: el modelo, entrenado con la máscara de saltos, proyecta la degradación y la intervención baja el voltaje ~0,5 V.
- El RMSE de la Bi-LSTM varía **±2 % entre semillas**, más que lo que falta para C1 a 60 días (0,4 %).

## Resultados

| Intento | Validación cruzada (train) | Validación (07) | Estado |
|---|---|---|---|
| Ensamble de 5 semillas | Mejora pequeña y estable (≈ −1 % RMSE) | — (incluido en la v2) | Adoptado en la v2 |
| Corrección por riesgo de intervención (λ = 0,05) | Las intervenciones se concentran en el 10 % de celdas de mayor voltaje (67 % de los saltos). Con λ = 1 multiplica el error en bloques sin campañas; λ = 0,05 mejora −2,3 % / −3,2 % / +0,3 % | Bi-LSTM v2: RMSE 1–180 0,0580 V (vs 0,0585), 60 d 0,0576 V (persistencia 0,0574), 180 d 0,0703 V (vs 0,0691). No cierra ningún criterio y pierde C4 a 180 d | Descartado como modelo elegido (en `main`, notebook 06e) |
| Entrenar con ventanas VMD **causales** (como en la inferencia) | 60 d peor en los 3 bloques (F1 0,1277 vs 0,1250; F2 0,0445 vs 0,0439; F3 0,0199 vs 0,0194); 180 d algo mejor | No evaluado | Rechazado |
| Combinación con la persistencia `w·modelo + (1 − w)·y(t0)` | Ganancia ≤ 0,9 % a 60 d (dentro del ruido entre semillas); el peso óptimo por horizonte cambia de signo entre bloques | No evaluado | Rechazado |

## Conclusión

Los cambios de modelado que se pueden justificar sin mirar la validación dan mejoras del orden de 1–2 %, el mismo orden que el ruido entre semillas. Lo que limita es la **información**, no la arquitectura: el momento de las intervenciones de mantenimiento no está en los datos. Para avanzar hace falta el **registro de mantenimientos** (fechas y celdas intervenidas) o el **plan de campañas**, que permitiría usar el riesgo de intervención con su fecha y no solo con el percentil de voltaje de la celda.
