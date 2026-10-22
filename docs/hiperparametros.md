# Fundamentación de configuraciones e hiperparámetros

**Objetivo:** que **cada valor** de configuración de la PoC tenga un fundamento verificable (una fuente, una medición o un análisis de los datos) y que ninguno sea una suposición o un criterio propio sin respaldo.

**Estado:** documento de análisis y plan. Los cambios de código y los reentrenamientos todavía **no se aplicaron**: se harán en la rama `feature/test`, en el orden de la sección 9.

## Marcas

| Marca | Significado |
|---|---|
| **F** | Valor tomado de una fuente del Benchmarking o de la especificación, con artículo y tabla o sección. |
| **A** | Adaptación justificada con una medición, un análisis de nuestros datos o un motivo verificable. |
| **Prueba** | Se fundamenta con una prueba corta, con datos de train y sin usar la validación. |
| **Sin fundamento** | Seguiría sin fuente ni medición. Se declara como límite. |

## Contexto que motiva las adaptaciones

- **Datos:** voltaje **diario** de 300 celdas (150 por electrolizador), ~1 945 días de train. Las fuentes usan datos cada 5 minutos a 1 hora.
- **Tarea:** horizonte de 1 a 180 días, frente a 1 a 48 pasos en las fuentes.
- **Hardware:** GPU RTX 4060 de 8 GB.

---

## 1. Fuentes revisadas

Los PDF de ScienceDirect están en `TP202620029/Fuentes - Benchmarking` (Makri, Khan, Song, Zhang). Los demás son de acceso abierto: Palan en Europe PMC (PMC13183939), Duan en nature.com, Morcillo en Springer, Guan en SpringerOpen y Gou en Europe PMC (PMC10959395).

| Prop. | Fuente | Método de búsqueda | Presupuesto | Espacio de búsqueda | Óptimo publicado |
|---|---|---|---|---|---|
| A | Makri et al. (2026), *Results in Engineering* 29 | SOMA (compara con Hyperband y PSO) | Población 6, 100 migraciones, *path length* 4, paso 0,11 (Tabla 5) | Capas {1, 2, 3}; unidades {20, 40, …, 180}; dropout {0–0,5}; LR {1e-4, 1e-3, 3e-3, 1e-2}; épocas {8, 16, 32, 64}; lote {50, 100}; pérdida {MSE, MAE}; optimizador {Adam, Adadelta} (Tabla 1) | 3 capas, 51 unidades, dropout 0, LR 3e-3, 150 épocas, lote 16 (Tabla 6). Modelo base: 64 unidades, 1 capa, dropout 0, LR 1e-3, 50 épocas, lote 32 (Tabla 3) |
| A | Makri, BO-VMD | Optimización bayesiana con proceso gaussiano (scikit-optimize) | 30 evaluaciones | K ∈ [2, 15], α ∈ [400, 2000]; objetivo: error de reconstrucción (ec. 14) | K = 15, α = 500 |
| A | Makri, RFR-SHAP | — | — | — | Las 12 variables más importantes (umbral fijo) |
| A | Makri, división de datos | — | — | — | 90 % entrenamiento y validación, 10 % prueba |
| B | Khan et al. (2026), *Energy Reports* 15 | ATPE (TPE adaptativo). En la Tabla 9 lo comparan con grilla y búsqueda aleatoria y gana | 100 trials por modelo; corte tras 20 trials sin mejora (sec. 3.7) | LR [1e-5, 1e-2] logarítmico; lote {16, 32, 64, 128}; dropout [0,1; 0,5]; BiLSTM capa 1 [128, 512] y capa 2 [64, 256]; densas [128, 512] y [64, 256]; Kalman q y R [1e-6, 1e-2]; α de fusión [0,3; 0,7] (Tabla 10) | BiLSTM (200, 100), densas (256, 128), secuencia 24, lote 32, Adam 1e-3, dropout 0,2, L2 1e-4, 200 épocas con paciencia 20, CV de 5 particiones (Tabla 7); α* = 0,58–0,62 (Tabla 8); Kalman con estado [nivel, velocidad, aceleración] |
| B | Song et al. (2025), *Energy and AI* 20 | Optuna (TPE) | No indicado | No publicado para la red | CNN de 128 filtros, 2 BiLSTM de 128, dropout 0,3, L2 0,01, LR 1e-4, lote 64, paciencia 20, máximo 100 épocas (sec. 3.2). Split cronológico: 2 años de entrenamiento (20 % para validación) y 1 año de prueba |
| B | Gou et al. (2024), *PLOS ONE* 19 (citado por Khan: Seq2Seq BiLSTM con atención multi-cabeza) | No indicado | — | — | Encoder BiLSTM de 32, decoder de 64, 3 capas BiLSTM, LR 1e-3. **No publica el número de cabezas de atención** |
| C | Palan & N (2026), *Scientific Reports* 16 | Optuna | 20 trials | Unidades {64, 128, 256, 512}; capas {1–4}; LR [2e-4, 5e-3] logarítmico; épocas 70–130 (Tabla 3) | 512 unidades, 2 capas, dropout 0,2 (fijo), LR 7,69e-4, 90 épocas; Adam con weight decay (sin valor publicado) y early stopping. Train: primer 70 % (Tabla 2). **"Espacial" significa relaciones entre las variables de entrada:** una GRU espacial recorre el vector de variables de cada instante y luego una GRU temporal recorre los instantes. No usa un grafo físico |
| D | Morcillo-Jimenez et al. (2024), *Applied Intelligence* 54 | Grid search | 24 combinaciones | Capas {1, 2}; dropout {0,15; 0,3; 0,5}; neuronas {16, 32, 64, 128} (Tabla 2) | Seq2Seq: 2 capas, 64 neuronas, dropout 0,3 (Tabla 3) |
| D | Duan et al. (2026), *npj Adv. Manufacturing* 3 | Ninguno (configuración fija) | — | — | TCN de 3 capas convolucionales (32, 64 y 128 filtros), kernel 3, LR 1e-3, lote 32 |
| D | Palan & N (2026), TCN de comparación | Fija | — | — | 64 canales, kernel 3, 4 capas, dropout 0,2 (Tabla 4) |
| E | Guan Min (2026), *Energy Informatics* 9 | Algoritmo genético | 50 generaciones × 30 individuos | Filtros {16, 32, 64}; kernel {3, 5, 7}; unidades BiLSTM {32, 64, 128}; LR {1e-4, 1e-3, 1e-2} | No revisado |
| E | Zhang et al. (2026), *ECM: X* 31 | Pruebas preliminares; "no se hizo búsqueda automática" | — | — | LSTM 2 × 128, dropout 0,1, LR 1e-3, lote 256, 100 épocas |

**Conclusiones de la revisión:**
- 6 de las 8 fuentes del Benchmarking usan una búsqueda automática; ninguna prueba todas las combinaciones.
- Khan (Tabla 9) muestra que, con el mismo presupuesto de 100 trials, la búsqueda tipo TPE supera a la grilla y a la búsqueda aleatoria (RMSE 0,642 frente a 0,649 y 0,653). Optuna usa TPE por defecto, igual que Song y Palan.
- El "óptimo absoluto" no existe en la práctica: entre semillas el RMSE de la Bi-LSTM ya varía ±2 % (notebook 06e). Lo que se busca es una configuración que no se pueda mejorar de forma medible.

## 2. Hallazgos de la auditoría de la configuración inicial

Son problemas de la configuración con la que se entrenaron los modelos hasta ahora:

1. **El SOMA de la Propuesta A no reprodujo el de Makri.** Usamos población 5, 3 migraciones, *path length* 2, paso 0,4 y PRT 0,3, y optimizamos 3 hiperparámetros. Makri usa población 6, 100 migraciones, *path length* 4 y paso 0,11, y optimiza 8.
   - **Costo de la configuración de Makri:** cada migración mueve 5 individuos durante 4 ÷ 0,11 ≈ 36 pasos, unas 180 evaluaciones; con 100 migraciones son **~18 000 evaluaciones**.
   - Con ~1,2 min por evaluación (A1: 73 min en 61 evaluaciones), eso equivale a **~15 días por variante y ~30 días para A1 y A2**. El SOMA reducido tomó ~3 h.
   - Los valores reducidos no tenían fuente.
2. **El BO-VMD de A tampoco sigue a Makri:** usamos TPE en lugar de proceso gaussiano, K [3, 10] en lugar de [2, 15], α [500, 5000] en lugar de [400, 2000], y entropía de envolvente en lugar de error de reconstrucción. **Cambiarlo afecta a todos los modelos que usan la señal VMD, incluidos los nuestros.**
3. **El componente espacial del ST-GRU (C) no coincide con Palan:** usamos un GCN sobre un grafo de celdas (2 vecinas en serie y 4 de mayor correlación), que era nuestra interpretación del texto del Benchmarking.
4. **Dos óptimos terminaron en el borde de su rango:** A2 (LR = 0,01 y 256 unidades) y el α de fusión de B (0,70). Esos espacios quedaron cortos.
5. **Makri tiene una inconsistencia interna:** su óptimo (51 unidades, 150 épocas, lote 16) está fuera de su propio espacio de búsqueda (Tabla 1).
6. **La configuración inicial de nuestra Bi-LSTM se fijó sin búsqueda**, con valores habituales. Sin embargo, la mayoría de sus valores coincide con valores publicados (sección 4).

---

## 3. Criterio para fundamentar y por qué baja el tiempo

**Camino 1 (adoptado): usar el óptimo que publica cada fuente.**
- **Qué es:** el autor ya hizo su búsqueda de hiperparámetros y publicó el resultado; nosotros adoptamos esa configuración y entrenamos cada modelo **una sola vez**.
- **Cómo se fundamenta:** cada valor se justifica citando la tabla del artículo. Por ejemplo: "La Propuesta A usa 3 capas y 51 unidades porque es la configuración que Makri et al. (2026) obtuvieron con SOMA (Tabla 6)". Es el mismo criterio con el que se construyó el Benchmarking: implementar cada propuesta tal cual la literatura.
- **Costo:** 3–4 h de cómputo, más ~1,5 h de pruebas cortas (sección 8).
- **Limitación que se declara en la memoria:** cada óptimo se encontró con los datos del autor (consumo cada 10 min, iluminación horaria, etc.), no con los nuestros, así que puede no ser el óptimo para nuestras celdas.

**Camino 2 (documentado como mejora posible, no se ejecuta): repetir la búsqueda de cada autor con nuestros datos.**
- **Qué es:** cada búsqueda entrena el modelo cientos o miles de veces (100 trials en Khan, ~18 000 evaluaciones en el SOMA de Makri).
- **Costo:** **40–55 h de GPU**, es decir, la placa de video trabajando de forma continua; la PC tiene que quedar encendida y sin suspenderse. Para SOMA haría falta, además, una regla de parada: cortar cuando el líder no mejora en una migración completa, que Makri admite en el paso 6 de su sec. 2.4.3 ("número máximo de migraciones o umbral de convergencia").
- **Qué resuelve:** la limitación del camino 1.

**Nuestra Bi-LSTM:** no hay fuente con su configuración, porque el modelo es nuestro. Pero casi todos sus valores coinciden con valores publicados (sección 4). Solo se buscan los que no aparecen en ninguna fuente: una grilla de 9 combinaciones reemplaza a una búsqueda de 100 trials. Una búsqueda amplia con Optuna queda como mejora posible, no como requisito para fundamentar.

**Por qué hay que reentrenar todo y volver a ejecutar el notebook 07:**
- Cambiar la configuración cambia el modelo, y los `.pt` guardados son los de la configuración anterior.
- El notebook 07 calcula a partir de esos modelos la comparación (tabla de los 11 modelos, ranking, Tabla 10 / RNF-06 y C1–C4), los CSV de `reports/` y el README.
- **Los resultados van a cambiar.** Los modelos del Benchmarking pueden mejorar con la configuración de su autor; si nuestro modelo sigue primero, la conclusión queda más sólida, y si alguno lo supera, se reporta tal cual.

---

## 4. Parámetros de cada modelo: valor inicial y cómo quedan

### 4.1 Bi-LSTM y Bi-NOA-LSTM (modelos propios)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Unidades / capas | 64 / 1 | Igual: modelo base de Makri (Tabla 3) | F |
| Learning rate | 1e-3 | Igual: Makri (Tabla 3) y Khan (Tabla 7) | F |
| Dropout | 0,2 | Igual: Khan (Tabla 7) | F |
| Weight decay | 1e-4 | Igual: Khan, L2 = 1e-4 | F |
| Capa densa de salida | 256 | Igual: Khan, primera capa densa (Tabla 7) | F |
| Paciencia del early stopping | 10 | 20: Khan y Song | F |
| Máximo de épocas | 80 | 200: Khan | F |
| Recorte de gradiente | 1,0 | Se elimina: ninguna fuente lo usa | F |
| Lote | 1024 | Igual, justificado con una medición de tiempo por época y memoria de GPU con lotes de 32, 128 y 1024 | A (medición) |
| Orígenes de train cada 5 días | 5 | Igual, justificado con una medición del error con orígenes cada 1 y cada 5 días | A (medición) |
| Embedding de celda | 8 | Grilla {4, 8, 16} | Prueba |
| Ancla (días promediados) | 7 | Grilla {3, 7, 14}. Una semana no se justifica por un ciclo semanal: RFR-SHAP no seleccionó el día de la semana (importancia < 5 %) | Prueba |
| Bloque de early stopping | últimos 270 días de train | Igual: ~15 % de train. **Pendiente:** confirmar en las fuentes la proporción de validación dentro de train (Song usa 20 %; Makri no la detalla) | A (pendiente de confirmar) |
| Máscara de saltos | sí | Igual: ablación del notebook 06 | A |
| Celda LSTM frente a NOA | LSTM | Igual: ablación de los notebooks 06 y 07 | A |
| Bloques de la validación cruzada | F1, F2, F3 | Igual: elegidos para incluir períodos con campañas de mantenimiento (F1, F2) y sin ellas (F3) | A |

**Bi-LSTM v2 (notebook 06e): intento descartado.** No es mejor que la Bi-LSTM original y no cumple ningún criterio que la original no cumpla, así que **se quita de la comparación de la PoC** (notebook 07, ranking, README y tabla de modelos). El notebook 06e y `docs/intentos_mejora.md` se mantienen como registro del intento, porque respaldan que el límite son los datos y no el modelo. Sus parámetros (ensamble de 5 semillas, λ = 0,05, tramos de percentil) ya no forman parte de la PoC y no requieren fundamentación; la sección 6.1 queda solo como registro.

### 4.2 Propuesta A (Makri)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Arquitectura MIMO, Bi-NOA-LSTM y Min-Max | — | Igual | F |
| BO-VMD: algoritmo | TPE (Optuna) | Proceso gaussiano | F |
| BO-VMD: rangos | K [3, 10]; α [500, 5000] | K [2, 15]; α [400, 2000] | F |
| BO-VMD: objetivo | entropía de envolvente | error de reconstrucción (ec. 14) | F |
| BO-VMD: evaluaciones | 30 | Igual | F |
| SOMA | población 5, 3 migraciones, 3 hiperparámetros | **No se vuelve a correr:** se adopta el óptimo que Makri obtuvo con SOMA | F |
| Capas / unidades / LR | lo que halló nuestro SOMA (A1: 2 / 176 / 4,4e-3; A2: 1 / 256 / 1e-2) | 3 / 51 / 3e-3 (Tabla 6) | F |
| Dropout | 0,2 | 0 (Tabla 6) | F |
| Lote / épocas | 32 / early stopping (80, paciencia 10) | 16 / 150 fijas (Tabla 6) | F |
| Weight decay | 1e-4 | Se elimina: Makri no lo usa | F |
| Recorte de gradiente | 1,0 | Se elimina: Makri no lo usa | F |
| RFR-SHAP: selección | importancia ≥ 5 % | Igual: Makri toma las 12 más importantes, pero tenemos solo 4 candidatas (kA, voltaje medio de celda, día del año y día de la semana), así que "top-12" las incluiría a todas. Con el 5 % se seleccionaron kA y el día del año | A |

### 4.3 Propuesta B (Khan, Song, Gou)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Unidades del encoder / decoder | 64 / 128 | 32 / 64: Gou et al. (2024), la fuente de la arquitectura Seq2Seq con atención que cita Khan | F |
| Capas densas | — | 256 / 128: Khan (Tabla 7) | F |
| LR / dropout / regularización | 1e-3 / 0,2 / 1e-4 | Igual: Khan (Tabla 7) | F |
| Paciencia / máximo de épocas | 6 / 40 | 20 / 200: Khan | F |
| Ensamble de 3 semillas | 3 | Se elimina: Khan entrena un solo modelo | F |
| Kalman: estado | nivel y pendiente | nivel, velocidad y aceleración: Khan | F |
| Kalman: q y R | máxima verosimilitud | Igual. **Pendiente:** citar un libro de referencia de modelos de espacio de estados (Khan las ajusta con ATPE) | A (pendiente de cita) |
| α de fusión | grilla 0,30–0,70 | Igual: rango de Khan (Tabla 10) | F |
| Lote | 512 | Igual, justificado con una medición de memoria de GPU (decoder de 180 pasos) | A (medición) |
| Orígenes de train cada 10 días | 10 | Igual, justificado con una medición de tiempo por época | A (medición) |
| Embedding de celda | 8 | El valor que elija la grilla de nuestra Bi-LSTM, para mantener las mismas condiciones | A |
| Cabezas de atención | 4 | Ni Khan, ni Song, ni Gou publican el número | **Sin fundamento**, o Prueba con {1, 2, 4, 8} (~1 h) |

### 4.4 Propuesta C (Palan)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Módulo espacial | GCN sobre un grafo de celdas | GRU espacial que recorre las variables de cada día (las 150 celdas y la corriente en kA) y luego una GRU temporal, como describe Palan | F |
| Vecinas y celdas correlacionadas del grafo | 2 + 4 | Se eliminan junto con el grafo | — |
| Unidades / capas / LR / épocas | 64 / 1 / 1e-3 / early stopping | 512 / 2 / 7,69e-4 / 90 (Tabla 3) | F |
| Dropout | 0,2 | Igual (Tabla 3) | F |
| Weight decay | 1e-4 | 0,01, el valor por defecto de AdamW en PyTorch (Palan usa weight decay sin publicar el valor) | A |
| Lote | 16 | Palan no lo publica: se justifica con una medición de memoria de GPU | A (medición) |

### 4.5 Propuesta D (Duan, Palan)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Canales / kernel / dropout | 64 / 3 / 0,2 | Igual: Palan (Tabla 4) | F |
| Bloques | 5 (dilataciones 1, 2, 4, 8, 16) | 4: Palan (Tabla 4). Con kernel 3 cubren 61 días, más que la ventana de 30 | F |
| LR / lote | 1e-3 / 32 | Igual: Duan | F |
| Weight decay / recorte de gradiente | 1e-4 / 1,0 | Se eliminan: ninguna de las dos fuentes los usa | F |
| Épocas | early stopping (80, 10) | Early stopping con paciencia 20 (Khan). Las épocas de Duan (20 + 50 de ajuste fino) son de transferencia de aprendizaje, que aquí no se aplica | A |
| Búsqueda | ninguna | Ninguna: Duan no hace búsqueda. La grilla de Morcillo queda como mejora posible (camino 2) | F |

### 4.6 Métodos de referencia (baselines)

| Parámetro | Valor | Cómo queda | Marca |
|---|---|---|---|
| Persistencia | último valor real | Igual (especificación, `docs/prompt_v1.md`) | F |
| Tendencia lineal: ventana | 30 días | Igual: la ventana de entrada (sección 6.3) | F |
| ETS: modelo AAN y restricción β ≤ α | — | Igual: documentación de Microsoft y región admisible de ETS (Hyndman et al.) | F |
| ETS: grilla de α y β | 10 × 8 valores | **Sin fundamento.** Alternativa: optimizar α y β de forma continua, como hace ETS habitualmente | Sin fundamento / Prueba |
| ETS: ventana de validación | 182 días, cada 7 | Igual: coincide con el horizonte de 180 días y con la separación de orígenes de la evaluación | A |

### 4.7 Parámetros comunes del VMD causal

| Parámetro | Valor | Marca |
|---|---|---|
| Historia del VMD causal | 365 días | **Sin fundamento escrito.** Se cierra midiendo su efecto o con una fuente |
| Segmento mínimo para aplicar VMD | 60 días | **Sin fundamento escrito** |
| Modos de alta frecuencia descartados | 1 | **Sin fundamento escrito** |

---

## 5. Lo que seguiría sin fundamento después del camino 1

1. **Cabezas de atención de B (4):** ninguna fuente las publica. Se puede cerrar con una prueba de 1, 2, 4 y 8 cabezas (~1 h).
2. **Grilla de α y β de ETS:** se puede reemplazar por una optimización continua.
3. **Parámetros del VMD causal** (365 días de historia, 60 días de segmento mínimo, 1 modo descartado): hay que medir su efecto o buscar una fuente.
4. **Umbrales de limpieza con respaldo parcial** (sección 6.2): interpolación de 3 días, tramos mínimos de 7 y 5 días, ventana de 7 días para saltos. Se cierran con pruebas cortas.
5. **Ventana de entrada de 30 días** (sección 6.3): **hoy no tiene fundamento propio**, porque se tomó de la especificación. Se cierra con la grilla de 7, 14, 30, 60 y 90 días y la regla de un error estándar (sección 8).

---

## 6. Verificaciones hechas con los datos

### 6.1 Tramos de percentil del riesgo de intervención (v2, intento descartado: solo registro)

Probabilidad de una caída de nivel en los 180 días siguientes según el percentil de voltaje de la celda en su electrolizador (train, orígenes cada 7 días):

| Percentil | Probabilidad de caída en 180 días |
|---|---|
| 0 – 70 % | 0,5 – 1,8 % (plana) |
| 70 – 75 % | 2,7 % |
| 75 – 80 % | 7,5 % |
| 80 – 85 % | 15,6 % |
| 85 – 90 % | 38,1 % |
| 90 – 95 % | 55,3 % |
| 95 – 100 % | 66,7 % |

- Los cortes en 0,8, 0,9 y 0,95 coinciden con los saltos de riesgo.
- **El corte en 0,5 no se justifica:** el riesgo es igual a ambos lados.
- **El tramo 0,8–0,9 mezcla dos riesgos muy distintos** (15,6 % y 38,1 %).
- Los tramos que respaldan los datos serían: 0–0,7 · 0,7–0,8 · 0,8–0,85 · 0,85–0,9 · 0,9–0,95 · 0,95–1.

**Decisión:** la v2 se descartó de la PoC (sección 4.1). Esta tabla queda como registro del análisis y como evidencia de que el riesgo de intervención depende del nivel relativo de la celda.

### 6.2 Umbrales de limpieza (Silver, notebooks 03 y 04)

El análisis exploratorio del notebook 03 justifica **por qué existe cada regla** (secciones 3, 4, 5.1 y 7, y la tabla de reglas R1–R9 de la conclusión). Estas verificaciones justifican **el valor numérico** de cada umbral. Usan el período completo, porque los umbrales se aplican antes de entrenar y no usan métricas de validación.

| Umbral | Qué muestran los datos | Conclusión |
|---|---|---|
| **V < 0,5 V con kA normal = fuera de servicio** (R5) | En operación **no hay ninguna** medición entre 0,5 y 2,5 V. El voltaje normal va de 2,86 a 3,77 V (percentiles 0,01 % y 99,99 %) | **Justificado.** Cualquier corte entre 0,5 y 2,5 V da el mismo resultado |
| **V > 5 V = atípico** (R6) | No hay ninguna medición en operación entre 4 y 5 V; el máximo normal es 3,77 V | **Justificado** |
| **Salto de nivel ≥ 0,10 V** (R9) | Sin intervenciones, la diferencia entre las medianas de 7 días antes y después es < 0,037 V en el 99 % de los casos y < 0,059 V en el 99,5 %. Por encima de +0,10 V queda solo el 0,03 % | **Justificado:** casi 3 veces la variación normal |
| **Paro: kA < 10** (R4) | Operación normal entre 12 y 16 kA (voltaje medio 3,12–3,18 V). Con 5–10 kA (78 días) el voltaje cae a 2,85 V, unos 3 desvíos por debajo. Entre 10 y 12 kA hay 3 días con 2,96 V, que también parecen anómalos | **Justificado, con un ajuste propuesto:** subir el corte a **12 kA**, que afecta solo a esos 3 días |
| **Interpolar huecos de hasta 3 días** (R3) | El 83 % de los huecos dura ≤ 3 días (718 de 915 duran 1 día), pero no hay un corte natural en 3 | **Parcial.** Prueba: ocultar valores conocidos, interpolarlos y medir desde qué largo de hueco el error supera la variación diaria |
| **Fuera de servicio: tramo ≥ 7 días e interrupciones ≤ 5 días** (R5) | Los tramos con V < 0,5 V tienen todas las duraciones, sin un corte claro en 7 | **Parcial.** Los tramos más cortos igual se descartan como atípicos o huecos, así que el umbral cambia la etiqueta pero no los datos que ve el modelo. Prueba: verificarlo |
| **Ventana de 7 días para detectar saltos** (R9) | No probado | **Pendiente.** Prueba: comparar 3, 5, 7 y 14 días |

### 6.3 Ventana de 30 días, horizonte de 180 días y split 80/20

La especificación (`docs/prompt_v1.md`) fija estos tres valores. Como eso solo no alcanza, se fundamentan con los datos y con las fuentes.

**Horizonte de 180 días: justificado (A). Estado: hecho.**
- **El horizonte es un requisito, no un hiperparámetro.** No se optimiza por error, porque el error siempre crece con el horizonte; se define por hasta dónde necesita ver el usuario.
- **De dónde sale el valor exacto:** el sistema tiene que anticipar qué celdas superarán el umbral antes de la próxima campaña de mantenimiento. Ese plazo de planificación se define como **6 meses**, que expresados en meses de 30 días son **180 días** (6 × 30).
- **Por qué no 181 u otro valor cercano:** el modelo pronostica cada día del 1 al 180, así que el usuario puede consultar cualquier plazo intermedio, y entre 178 y 181 días el error prácticamente no cambia (la curva de error frente al horizonte del notebook 07 es suave). Ninguna medición puede distinguir 180 de 181; lo que se justifica es el orden de magnitud (unos 6 meses) y la convención de expresarlo en meses de 30 días.
- **Respaldo con los datos:** en train hubo 16 intervalos entre campañas de mantenimiento: **mediana de 137 días**, **percentil 75 de ~178 días**, y **12 de 16 duran menos de 180 días**. Las campañas se detectaron como ≥ 5 saltos del electrolizador en 30 días; las del EL A fueron jul-20, oct-20, mar-21, jul-21, ene-22, may-22, oct-22, feb-23 y jul-24.
- Con un horizonte de 180 días, el ingeniero ve qué celdas superarán el umbral **antes de la próxima campaña**, que es lo que necesitan las alertas de las historias US026 y US028.

**Split 80/20: justificado (F + A).**
- **Fuentes:** la parte de prueba va del 10 % al 33 %. Makri usa 90/10, Palan 70/30 (Tabla 2) y Song ~67/33 (2 años de entrenamiento y 1 año de prueba). El 20 % está dentro de ese rango.
- **Datos:** la validación tiene que contener al menos una ventana completa de 30 + 180 = 210 días y suficientes orígenes para medir la significancia de C2.

| Split | Días de validación | Orígenes con el horizonte completo |
|---|---|---|
| 90/10 | 243 | ~10 |
| **80/20** | **486** | **44** |
| 70/30 | ~730 | más, pero train pierde ~8 meses de los datos más recientes |

80/20 es el mayor porcentaje de entrenamiento que mantiene una validación estadísticamente útil. El 90/10 se evaluó y se descartó (ver "Decisiones" en `prompt.md` y en el README).

**Ventana de 30 días: sin fundamento propio por ahora. Estado: pendiente.**
- **Cómo se eligió:** se tomó de la especificación original (`docs/prompt_v1.md`, L = 720 h = 30 días). **Ningún notebook de la PoC probó otra ventana ni aplicó un método para elegirla.**
- **Lo único que se verificó** (en el análisis de esta documentación, no en un notebook): la autocorrelación media de los cambios diarios del voltaje (train) es −0,18 a 1 día y ~0 desde los 2 días. Eso indica que la dinámica de corto plazo dura 1 o 2 días y que no hace falta una ventana muy corta, pero **no justifica el valor 30**: una ventana más larga podría estimar mejor el nivel y la tendencia.
- **Cómo se va a fundamentar** (prueba pendiente, sección 8): grilla de 7, 14, 30, 60 y 90 días en la validación cruzada de 3 bloques, con 3 semillas, eligiendo con la regla de un error estándar. Si la regla elige otro valor, se cambia la ventana de todos los modelos.

---

## 7. Espacio de búsqueda de la Bi-LSTM (camino 2, mejora posible)

No es necesario para fundamentar la configuración (sección 4.1), pero queda definido y con fuente por si se ejecuta la búsqueda amplia.

- **Método:** Optuna con TPE (Song, Palan; Khan, Tabla 9).
- **Presupuesto:** hasta 100 trials, con corte tras 20 sin mejora (Khan, sec. 3.7).
- **Validación:** la validación cruzada temporal de 3 bloques del notebook 06e. Khan usa CV de 5 particiones, pero en series de tiempo las particiones tienen que respetar el orden temporal (A).
- **Objetivo:** el promedio entre bloques del RMSE relativo; el R² sobre el cambio es criterio secundario.
- **Validación final:** la configuración ganadora se evalúa una sola vez en la validación del notebook 07, y reemplaza a la actual solo si mejora.

| Hiperparámetro | Rango | Escala | Marca | Fundamento |
|---|---|---|---|---|
| Learning rate | [1e-4, 1e-2] | logarítmica | F | Makri (Tabla 1) y Guan; incluye los óptimos de Palan (7,7e-4) y Khan (1e-3) |
| Unidades de la LSTM | [32, 256] | logarítmica | A | Makri {20–180}, Khan capa 2 [64, 256], Guan {32–128}. Se excluyen los 512 de Palan y de la capa 1 de Khan porque, con ~1 945 días por celda, un modelo mayor sobreajusta (riesgo 3 del Project Charter) |
| Capas | {1, 2, 3} | entera | F | Makri (Tabla 1); el óptimo de Palan es 2 |
| Capa densa de salida | [128, 512] | logarítmica | F | Khan, capa densa 1 (Tabla 10) |
| Dropout | [0, 0,5] | lineal | F | Makri (Tabla 1) y Morcillo; Khan [0,1; 0,5] |
| Regularización (weight decay) | [1e-6, 1e-3] | logarítmica | A | Khan (L2 1e-4) y Song (L2 1e-2); rango centrado en el óptimo de Khan |
| Lote | {256, 512, 1024, 2048} | categórica | A | Las fuentes usan 16–128 con miles de muestras; nosotros tenemos ~108 000 por época, y un lote de 16 haría cada época ~60 veces más lenta. Zhang usa 256 |
| Épocas | early stopping, paciencia 20, máximo 200 | — | F | Khan (200 / 20) y Song (100 / 20) |
| Ventana de entrada | {30, 60, 90} días | categórica | F / A | 30 días: especificación (L = 720 h). Guan y Palan también optimizan la ventana. Hasta 90 días para no perder demasiados orígenes de train |
| Ancla (días promediados) | [3, 14] | entera | Prueba | No está en las fuentes (contribución propia); va de casi el último valor a dos semanas, siempre menor que la ventana |
| Embedding de celda | [4, 16] | entera | Prueba | No está en las fuentes (contribución propia: pesos compartidos entre celdas) |

Costo estimado: ~6 h. Para los otros modelos, el camino 2 sería: A, SOMA con los parámetros y el espacio de Makri y la regla de parada (10–20 h por variante); B, búsqueda TPE con el espacio de Khan, incluidos q, R y α (~8 h); C, Optuna con el espacio de Palan, 20 trials (~2–3 h); D, la grilla de Morcillo (~0,5 h).

---

## 8. Pruebas cortas pendientes

**Estado: ninguna se ejecutó todavía.** Se hacen con datos de train y **antes** de reentrenar, porque si alguna cambia un valor (por ejemplo 12 kA o una ventana de 60 días), cambian los datos de entrada de todos los modelos.

**Regla para elegir un valor dentro de una grilla:** la **regla de un error estándar** (Breiman et al., 1984, *Classification and Regression Trees*; Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*, sec. 7.10):
1. Se mide el error de cada valor de la grilla en la validación cruzada de 3 bloques, con 3 semillas, para estimar su variación.
2. Se elige **el valor más simple** (ventana más corta, modelo más chico) **cuyo error esté dentro de un error estándar del mejor**.

La regla da un único valor y explica por qué, por ejemplo, se elige 30 y no 29: los valores fuera de la grilla no se prueban porque sus diferencias son menores que la variación entre semillas (±2 %), y entre los de la grilla gana el más simple que no es peor de forma medible. Se aplica a todas las grillas: ventana, ancla y embedding, cabezas de atención de B y ventana de detección de saltos.

| Prueba | Qué fundamenta | Tiempo |
|---|---|---|
| Interpolación: ocultar valores y medir el error según el largo del hueco | R3 (3 días) | minutos |
| Etiquetado fuera de servicio con tramos de 3, 5, 7 y 14 días | R5 (7 y 5 días) | minutos |
| Ventana de detección de saltos de 3, 5, 7 y 14 días | R9 (7 días) | minutos |
| Corte de paro en 10 y 12 kA | R4 | minutos |
| Ventana de entrada de 7, 14, 30, 60 y 90 días (validación cruzada de 3 bloques, 3 semillas, regla de un error estándar). El VMD causal se calcula una sola vez con 90 días y las ventanas más cortas se toman de ahí | L | ~1 h |
| Grilla ancla {3, 7, 14} × embedding {4, 8, 16} (regla de un error estándar) | Bi-LSTM | ~40 min |
| Mediciones de lote y separación de orígenes (Bi-LSTM, B, C) | Lote, orígenes cada 5 y cada 10 días | ~20 min |
| Cabezas de atención {1, 2, 4, 8} (opcional) | B | ~1 h |
| ETS con α y β continuos (opcional) | ETS | minutos |

---

## 9. Orden de trabajo (rama `feature/test`)

1. Pruebas cortas de la sección 8 y actualización de este documento con sus resultados.
2. Cambios de configuración y de código: BO-VMD de Makri, módulo espacial de Palan para C, Kalman de Khan para B, la configuración de la sección 4, y **quitar la v2 de la comparación** (notebook 07, ranking, README).
3. Reconstruir Silver y Gold (notebooks 04, 05 y 05b) si cambian umbrales o el BO-VMD.
4. Reentrenar todos los modelos (notebooks 06, 06b, 06c, 06d y 06e).
5. Ejecutar de nuevo el notebook 07, una sola vez, y actualizar el README, `docs/metricas.md` y `docs/intentos_mejora.md`.
6. Merge a `main`.

**Tiempo estimado:** 3–4 h de cómputo, más ~1,5 h de pruebas cortas.
