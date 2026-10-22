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
| RFR-SHAP: selección | importancia ≥ 5 % sobre 4 covariables | Las 12 más importantes entre los modos VMD y las covariables, como Makri (sección 4.2.1) | F |


### 4.2.1 Hallazgos al aplicar el BO-VMD y el RFR-SHAP de Makri

- **Makri no usa el VMD para filtrar.** Usa los **modos como variables de entrada**: descompone la serie que pronostica en 15 modos, los agrega como candidatos junto con las demás variables, y RFR-SHAP elige las 12 más importantes (secciones 3.2 y 3.3). No descarta ni suma modos. El uso del VMD como filtro (`N_DROP`) venía de la interpretación del documento del Benchmarking.
  - **Propuesta A:** se implementa como Makri: modos como candidatos y RFR-SHAP con 100 árboles (el valor que publica Makri) eligiendo las 12 más importantes.
  - **Nuestro modelo:** cómo usa la señal (sin filtrar, filtrada sin 1, 2 o 3 modos, o modos como entradas) se decide con una prueba de menor error (paso 3, etapa 1).
- **BO-VMD por celda** (proceso gaussiano con `GPSampler` de Optuna, que implementa el mismo método que scikit-optimize, la biblioteca de Makri; 30 evaluaciones, K ∈ [2, 15], α ∈ [400, 2000], error de reconstrucción): **298 de 300 celdas eligieron K = 15 y α = 400**; una eligió K = 13 y otra K = 15 con α = 513. Es el borde del rango, porque con más modos y bandas más anchas la suma reconstruye mejor la señal; Makri obtuvo lo mismo (K = 15, el máximo). Tomó 38 minutos.
- **Celdas sin tramo en servicio en train** (A_c123 y A_c124, puenteadas hasta el final de train): no hay datos para optimizar su VMD; se les asignan los parámetros más frecuentes (K = 15, α = 400) (A).

### 4.3 Propuesta B (Khan, Song, Gou)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Unidades del encoder / decoder | 64 / 128 | 32 / 64: Gou et al. (2024), la fuente de la arquitectura Seq2Seq con atención que cita Khan | F |
| Capas densas | — | **No se agregan.** Las capas densas de 256 y 128 de Khan (Tabla 7) son de su BiLSTM independiente, que B no usa; la red de B es la Seq2Seq de Gou, que da el pronóstico por una sola capa totalmente conectada ("output through the fully connected layer", ec. 14) | F |
| LR / dropout / regularización | 1e-3 / 0,2 / 1e-4 | Igual: Khan (Tabla 7) | F |
| Paciencia / máximo de épocas | 6 / 40 | 20 / 200: Khan | F |
| Ensamble de 3 semillas | 3 | Se elimina: Khan entrena un solo modelo | F |
| Kalman: estado y matriz de transición | nivel y pendiente | **Aplicado:** nivel, velocidad y aceleración, con A = [[1, 1, 0,5], [0, 1, 1], [0, 0, 1]] y Q = q·I (Khan, sec. 3.6) | F |
| Kalman: corrección con observaciones | — | Sobre la historia hasta t0, como Khan | F |
| Kalman: pronóstico de 180 días | nivel + pendiente·h | **Aplicado:** se proyecta A sin corregir (nivel + velocidad·h + ½·aceleración·h²). Khan pronostica un paso y corrige con la observación de ese paso; para los días futuros no hay observaciones | A |
| Kalman: entrada de control (predicción de la red) | — | **Se omite:** con B = [1, 0, 0] y sin observaciones que corrijan, sumaría cada día la predicción completa al nivel (Khan, ec. 12). Riesgo declarado: proyectar la aceleración 180 días puede dar valores extremos; el peso α lo atenúa | A |
| Kalman: q y R | máxima verosimilitud de 3 varianzas | Máxima verosimilitud de q (con Q = q·I, como Khan) y r. Los valores de Khan (Tabla 8) no se pueden adoptar porque están en kWh². **Pendiente:** citar un libro de referencia de modelos de espacio de estados | A (pendiente de cita) |
| α de fusión | grilla 0,30–0,70 con paso 0,05 | **Aplicado:** paso 0,01 en [0,3; 0,7], elegido en la validación interna, como Khan ("grid search, step 0.01", sec. 3.6) | F |
| Lote | 512 | Igual, justificado con una medición de memoria de GPU (decoder de 180 pasos) | A (medición) |
| Orígenes de train cada 10 días | 10 | Igual, justificado con una medición de tiempo por época | A (medición) |
| Embedding de celda | 8 | El valor que elija la grilla de nuestra Bi-LSTM, para mantener las mismas condiciones | A |
| Cabezas de atención | 4 | Ni Khan, ni Song, ni Gou publican el número | **Prueba** {1, 2, 4, 8}, junto con *teacher forcing* (sección 10.4.1). El código ya admite las dos opciones (`SEQ2SEQ["heads"]`, `SEQ2SEQ["teacher_forcing"]`) |

### 4.4 Propuesta C (Palan)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Módulo espacial | GCN sobre un grafo de celdas | **Aplicado:** GRU espacial que recorre las variables de cada día (las 150 celdas y la corriente en kA) y luego una GRU temporal; su último estado pasa por ReLU y una capa densa, como describe Palan. Un modelo por electrolizador, igual que D | F |
| Vecinas y celdas correlacionadas del grafo | 2 + 4 | Se eliminan junto con el grafo | — |
| Unidades / capas / LR / épocas | 64 / 1 / 1e-3 / early stopping | 512 / 2 / 7,69e-4 / 90 (Tabla 3; Tabla 4: 512 unidades y 2 capas para las dos GRU). Palan usa early stopping ("stops training when improvement is not being made within a specified patience period") sin publicar la paciencia: hasta 90 épocas con paciencia 20, la de Khan y Song, como en D | F (paciencia: A) |
| Dropout | 0,2 | Igual (Tabla 3) | F |
| Weight decay | 1e-4 | **1e-4.** Palan usa "Adam incorporating weight decay" sin publicar el valor. 1e-4 es la L2 de Khan (Tabla 7) y el máximo del rango que Palan busca para su BiLSTM-SSA ([1e-7, 1e-4], Tabla 4). (La versión anterior de este documento proponía 0,01, el valor por defecto de AdamW; no aplica, porque se usa Adam, cuyo valor por defecto es 0) | A |
| Lote | 16 | **64: Palan sí lo publica** ("DataLoaders take groups of 64 sequences"). Con 64 la red necesita 12,8 GB de GPU (medido; la GPU tiene 8 GB), así que cada lote se procesa en 4 partes de 16 y se acumulan los gradientes, ponderados por los objetivos de cada parte: el gradiente es el del lote de 64 (verificado con una prueba en `model.run_epoch`) y el pico de memoria baja a 3,3 GB | F (acumulación: A, medición) |

### 4.5 Propuesta D (Duan, Palan)

| Parámetro | Inicial | Cómo queda | Marca |
|---|---|---|---|
| Canales / kernel / dropout | 64 / 3 / 0,2 | Igual: Palan (Tabla 4) | F |
| Bloques | 5 (dilataciones 1, 2, 4, 8, 16) | 4: Palan (Tabla 4). Con kernel 3 cubren 61 días, más que la ventana de 30 | F |
| LR / lote | 1e-3 / 32 | Igual: Duan | F |
| Weight decay / recorte de gradiente | 1e-4 / 1,0 | Se eliminan: ninguna de las dos fuentes los usa | F |
| Épocas | early stopping (80, 10) | Early stopping con paciencia 20 y máximo de 200 épocas (Khan). Las épocas de Duan (20 + 50 de ajuste fino) son de transferencia de aprendizaje, que aquí no se aplica | A |
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
5. **Ventana de entrada de 30 días** (sección 6.3): **hoy no tiene fundamento propio**, porque se tomó de la especificación. Se cierra con la grilla de 7, 14, 30, 60 y 90 días, eligiendo la de menor error (sección 8). La prueba puede elegir otro valor.

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
| **Salto de nivel ≥ 0,10 V** (R9) | Sin intervenciones, la diferencia entre las medianas de 7 días antes y después es < 0,037 V en el 99 % de los casos y < 0,059 V en el 99,5 %. Por encima de +0,10 V queda solo el 0,03 % | **Análisis exploratorio, no decisorio:** el argumento usa percentiles elegidos por criterio propio. Se decide junto con la ventana de 7 días en la prueba de menor error del paso 3 |
| **Paro: kA < 10** (R4) | El voltaje de las celdas baja de forma lineal con la corriente, ~0,046 V por kA (R² = 0,95 en 138 días de 0,5 a 14 kA), y con ≤ 0,5 kA (apagado) se desploma ~2 V. Ejemplo: el 10-05-2022 el EL A operó con 10,5 kA y la mediana de sus celdas fue 0,18 V menor que la de los días vecinos (14,2 kA). Con el corte actual de 10 kA, ese día cuenta como válido | **Análisis exploratorio, no decisorio.** La comparación con días vecinos usaba criterios propios (percentil 99,5, ventana de ±7 días, referencia ≥ 14 kA). Se decide con la prueba de menor error (sección 8) entre 10 y 13,5 kA |
| **Valores interpolados: solo como entrada** | Un valor interpolado es una recta trazada entre dos mediciones, no una medición | **Decisión:** se usan solo como entrada del modelo (ventanas sin agujeros). El entrenamiento (objetivos) y las métricas usan **solo voltajes medidos** (calidad `ok`), como pide `docs/metricas.md` ("contra el voltaje real") |
| **Interpolar huecos de hasta 3 días** (R3) | Al ocultar valores conocidos de train e interpolarlos, el error es menor que el cambio medio de un día al siguiente (0,00258 V) hasta huecos de 12 días (0,00253 V), y lo supera desde 13 (0,00260 V). Pasar a 12 días agrega ~11 200 días-celda válidos (~2 %) | **Análisis exploratorio, no decisorio.** El criterio "no peor que el cambio diario" era propio. Se decide con la prueba de menor error (sección 8) entre 3 y 12 días. Pendiente: si los valores interpolados cuentan o no como objetivos y en las métricas |
| **Fuera de servicio: tramo ≥ 7 días e interrupciones ≤ 5 días** (R5) | Con mínimos de 3, 5, 7 y 14 días y uniones de 3, 5 y 7 días, los días-celda válidos de train son **idénticos** (547 507; 0 de diferencia) | **Justificado (A): insensible.** El umbral solo cambia la etiqueta entre categorías que igual se descartan |
| **Ventana de 7 días para detectar saltos** (R9) | No probado | **Pendiente.** Prueba: comparar 3, 5, 7 y 14 días |


#### Resultado de la prueba de menor error para R3 y R4 (hecha)

**Procedimiento (fijado antes de ejecutar):**
- **Variantes:** paro en 10 o 13,5 kA, por interpolación de 3 o 12 días.
- **Datos:** para cada variante se reconstruyeron Silver y Gold de train. El VMD usó los parámetros actuales, iguales en todas, para que solo cambie la limpieza.
- **Modelo:** la Bi-LSTM con la configuración actual (45 épocas, semillas 42, 43 y 44), en los bloques F1, F2 y F3.
- **Objetivos:** solo voltajes medidos.
- **Evaluación:** sobre los **mismos objetivos en todas las variantes** (los días-celda válidos en las cuatro: 2,66 M en F1, 2,88 M en F2 y 1,27 M en F3).
- **Criterio:** menor RMSE (1–180 días), relativo a la variante actual en cada bloque y promediado en los 3 bloques. Se informa el error estándar entre semillas.

| Paro | Interpolación | RMSE relativo (media ± error estándar) | RMSE F1 / F2 / F3 (V) |
|---|---|---|---|
| 10 kA | 3 días (actual) | 1,0000 | 0,1500 / 0,0453 / 0,0172 |
| **10 kA** | **12 días** | **0,9978 ± 0,0038** | 0,1494 / 0,0460 / 0,0169 |
| 13,5 kA | 3 días | 1,0102 ± 0,0055 | 0,1515 / 0,0457 / 0,0174 |
| 13,5 kA | 12 días | 1,0066 ± 0,0063 | 0,1523 / 0,0459 / 0,0170 |

**Decisiones:**
- **R4 (paro): se mantienen 10 kA.** Excluir los días de carga reducida (13,5 kA) no mejora el pronóstico: empeora el RMSE un 0,7–1,0 % en promedio.
- **R3 (interpolación): 12 días**, la variante de menor error.
- **Con estos datos, 3 y 12 días son equivalentes:** la diferencia (−0,22 %) es menor que el error estándar (0,38 %).
- **Efecto colateral:** con 12 días se detectan más saltos de nivel (443 frente a 368 en train), porque la regla R9 tiene datos donde antes había huecos. La prueba ya incluye ese efecto.
- **El análisis exploratorio anterior** (desvío frente a días vecinos, error de interpolación frente al cambio diario) queda solo como descripción de los datos, no como fundamento de la decisión.

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
- **Cómo se va a fundamentar** (prueba pendiente, sección 8): grilla de 7, 14, 30, 60 y 90 días en la validación cruzada de 3 bloques, con 3 semillas, eligiendo la ventana de menor error (criterio de las fuentes, sección 8). **Estado: en curso** (sección 6.4). **Decisión:** la ventana elegida se aplica solo a la Bi-LSTM propia; los modelos del Benchmarking mantienen 30 días y probar su ventana queda como mejora (sección 11, #6).

### 6.4 Pruebas del paso 3 (señal filtrada)

**Procedimiento común (fijado antes de ejecutar):**
- **Datos:** reconstruidos desde Silver, con la limpieza definitiva (10 kA, 12 días) y el BO-VMD de Makri por celda.
- **Modelo:** la Bi-LSTM con la arquitectura definitiva (Adam, ReLU, sin recorte de gradiente), 45 épocas fijas, semillas 42, 43 y 44, en los bloques F1, F2 y F3 de train.
- **VMD:** modos de train no causales; modos de los bloques causales.
- **Objetivos:** solo voltajes medidos, comunes a todas las variantes de cada etapa.
- **Criterio:** menor RMSE (1–180 días), relativo a la referencia y promediado en los 3 bloques. Se informa el error estándar entre semillas.

**Etapa 1: señal de entrada** (referencia: filtrada sin 1 modo, la configuración anterior)

| Señal | RMSE relativo (± error estándar) |
|---|---|
| Filtrada sin 1 modo | 1,0000 |
| Filtrada sin 2 modos | 1,0015 ± 0,0016 |
| **Filtrada sin 3 modos** | **0,9979 ± 0,0013** |
| Filtrada sin 4 modos | 0,9991 ± 0,0012 |
| Filtrada sin 5 modos | 1,0029 ± 0,0013 |
| Filtrada sin 6 modos | 1,0028 ± 0,0022 |
| Sin filtrar | 1,0267 ± 0,0051 |
| Modos VMD como entradas (como Makri) | 1,0702 ± 0,0059 |

**Decisiones:**
- **El filtrado VMD queda justificado** (A): sin filtro el error sube un 2,7 %, unas 5 veces el error estándar.
- **Los modos como entradas**, que es lo que hace Makri, empeoran un 7 % en nuestro modelo, así que no se adoptan para la Bi-LSTM (sí se usan en la Propuesta A, como su fuente).
- **`N_DROP = 3`**, el menor error. Es equivalente a 4 (diferencia del 0,12 %, menor que el error estándar). Descartar entre 1 y 6 modos cambia el error menos del 0,5 %.
- La grilla se extendió de 3 a 6 modos porque el ganador inicial (3) estaba en el borde.

**Etapa 2: ventana de entrada y ancla** (señal filtrada sin 3 modos)

Primero se probó la ventana con el ancla actual (7 días); después, como la ventana y el ancla dependen entre sí (el ancla promedia los últimos días de la ventana), una grilla conjunta, y por último la extensión del ancla, porque el ganador quedó en el borde.

| Ventana / ancla | RMSE relativo a 7 / 7 (± error estándar) |
|---|---|
| 90 / 7 | 1,0299 (frente a 30 / 7) |
| 60 / 7 | 1,0159 (frente a 30 / 7) |
| 30 / 7 (configuración anterior) | 1,0034 |
| 14 / 14 | 1,0066 ± 0,0033 |
| 14 / 3 | 1,0069 ± 0,0053 |
| 14 / 7 | 1,0019 ± 0,0014 |
| 7 / 7 | 1,0000 |
| 7 / 3 | 0,9982 ± 0,0042 |
| **5 / 3** | **0,9969 ± 0,0037** |
| 5 / 2 | 1,0029 (frente a 7 / 7) |
| 5 / 1 | 1,0035 (frente a 7 / 7) |
| 3 / 3 | 1,0141 ± 0,0055 |

(La fila 30 / 7 se calculó frente a 7 / 7 a partir de las dos comparaciones con objetivos comunes; las de 60 y 90 días son frente a 30 / 7.)

**Decisiones:**
- **Ventana de 5 días y ancla de 3 días** para la Bi-LSTM propia: el menor error.
- **Con estos datos, 5 / 3 es equivalente a 7 / 3 y a 7 / 7**: las diferencias quedan dentro del error estándar.
- **Las ventanas largas empeoran** (60 días: +1,6 %; 90 días: +3,0 %). Es coherente con la autocorrelación de los cambios diarios, que es ~0 desde los 2 días.
- **El ancla de 3 días es interior**: 1 y 2 días empeoran, igual que 7 y 14.
- **Los modelos del Benchmarking mantienen 30 días** (decisión de la sección 6.3); probar su ventana queda como mejora (sección 11, #6).


**Etapa 3: resto de la configuración de la Bi-LSTM** (base: ventana 5, ancla 3, señal sin 3 modos)

Se cambió **un factor por vez** respecto de la base; cuando el ganador quedó en el borde de su grilla, la grilla se extendió. Al final se verificó la combinación de los ganadores.

| Factor | Valores probados (RMSE relativo a la base, ± error estándar) | Decisión |
|---|---|---|
| Embedding de celda | 4: 1,0044 ± 0,0075 · **8: 1,0000** · 16: 1,0060 ± 0,0055 | **8** (interior) |
| Ventana de detección de saltos (R9) | 3: 1,0095 ± 0,0058 · 5: 1,0012 ± 0,0009 · **7: 1,0000** · 14: 1,0064 ± 0,0022 | **7 días** (interior) |
| Umbral de saltos (R9) | 0,02: 1,0073 ± 0,0044 · **0,03: 0,9910 ± 0,0010** · 0,05: 0,9969 ± 0,0013 · 0,10: 1,0000 · 0,15: 1,0011 ± 0,0020 | **0,03 V** (interior) |
| Lote | 512: 1,0076 ± 0,0029 · **1024: 1,0000** · 2048: 1,0020 ± 0,0047 | **1024** (interior). Tiempo por configuración: 355, 185 y 115 s |
| Orígenes de train | **cada 5 días: 1,0000** · cada 1 día con las mismas actualizaciones (9 épocas): 1,0063 ± 0,0053 · cada 1 día con 45 épocas: 1,0185 ± 0,0068 | **Cada 5 días**; además cuesta 5 veces menos |
| Recorte de covariables (\|z\|) | 1: 1,0021 ± 0,0010 · 2: 0,9946 ± 0,0037 · 3: 0,9966 ± 0,0025 · **5: 1,0000** · sin recorte: 1,0034 ± 0,0032 | **5.** Con el umbral de saltos definitivo (0,03 V), recortar en 2 da exactamente lo mismo que en 5 (1,0000 ± 0,0020); ante un empate se mantiene el valor actual. Sin recorte el error es mayor |
| Canal de máscara | sin: 1,0025 ± 0,0003 | **Se mantiene** |
| Indicador de electrolizador | sin: 1,0007 ± 0,0023 | **Se mantiene** (menor error; equivalente) |
| kA como entrada | sin: 1,1143 ± 0,0191 | **Se mantiene** (+11 % sin ella) |
| V_total / celdas en servicio como entrada | sin: 1,0552 ± 0,0039 | **Se mantiene** (+5,5 %) |
| Objetivo relativo al ancla | absoluto (respecto del voltaje medio de train, 3,156 V): 1,0474 ± 0,0047 | **Relativo al ancla** (+4,7 % con absoluto) |

**Configuración de la Bi-LSTM que surge del paso 3:** señal VMD sin los 3 modos de mayor frecuencia, ventana de 5 días, ancla de 3 días, embedding de 8, saltos de nivel para la máscara con umbral de 0,03 V y ventana de 7 días, recorte de covariables a \|z\| ≤ 5, lote de 1024 y orígenes cada 5 días. Frente a la configuración anterior del banco (ventana 30, ancla 7, sin 1 modo, umbral 0,10), el RMSE baja ~1,7 % (producto de las mejoras de cada etapa: 0,9979 × 0,9966 × 0,9969 × 0,9910 ≈ 0,9825; es aproximado, porque cada etapa se mide con sus propios objetivos comunes).

**Sobre el umbral de 0,03 V:** está por debajo de la variación normal de 7 días (el 99 % de las diferencias es menor que 0,037 V), así que la máscara de entrenamiento también marca variaciones chicas, no solo intervenciones. Se adopta porque es el de menor error. Su efecto se limita a la máscara de entrenamiento: no cambia las entradas, los objetivos de evaluación ni las métricas. **Decisión:** la regla R9 cumple dos funciones con umbrales distintos. (1) **Máscara de entrenamiento: 0,03 V** (menor error). (2) **Diagnóstico de la evaluación** (separar los objetivos que cruzan una intervención): **0,10 V**. Acá no aplica el criterio de menor error, porque no es una predicción, y 0,03 V marcaría como intervenciones variaciones que están dentro del ruido normal. 0,10 V se justifica con el análisis de los datos (sección 6.2): sin intervenciones, el 99 % de las diferencias de mediana a 7 días es menor que 0,037 V y solo el 0,03 % supera +0,10 V. **Sensibilidad (hecha, en los bloques de train):** con 0,07, 0,10 y 0,15 V cambia la porción marcada como "cruza una intervención" (F1: 6,8 / 6,5 / 6,2 %; F2: 2,6 / 2,4 / 1,7 %; F3: 0,7 / 0,3 / 0,15 %), pero la comparación Bi-LSTM frente a persistencia en cada subconjunto es la misma con los tres umbrales. **0,10 V queda justificado como insensible** dentro de la zona de transición entre el ruido (hasta ~0,06 V) y las intervenciones (desde ~0,2 V).

**Limitación del banco de pruebas:** todas las configuraciones se entrenan 45 épocas fijas (las que eligió el early stopping de la configuración anterior). Una configuración podría necesitar más o menos épocas; en el entrenamiento final, las épocas se eligen con early stopping (paciencia 20).

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

Costo estimado: ~6 h. La lista completa de mejoras del camino 2 está en la sección 11.

---

## 8. Pruebas cortas pendientes

**Estado: las de limpieza (R3, R4, R5) están hechas (sección 6.2) y las del paso 3 se registran en la sección 6.4 a medida que terminan.** Se hacen con datos de train y **antes** de reentrenar, porque cambian los datos de entrada de los modelos.

**Criterio para elegir un valor dentro de una grilla: el menor error, como en las fuentes.** Las 8 fuentes del Benchmarking eligen su configuración con el menor error de validación (Palan: "mejor pérdida de validación, minimizada en 20 trials", Tabla 3; Makri: menor MSE con SOMA; Khan: menor RMSE; Morcillo: mejor configuración de la grilla; Guan: MSE de validación como aptitud). Ninguna usa la regla de un error estándar, así que no se usa.
1. Cada valor de la grilla se mide en la validación cruzada de 3 bloques, con 3 semillas.
2. **Se elige el valor con menor error promedio.**
3. Se informa el error estándar de cada valor. Si el ganador y el segundo quedan a menos de ese margen, se deja escrito que son equivalentes con estos datos (igual se elige el de menor error).

**La prueba elige el valor; no está pensada para confirmar el actual.** Si gana otro valor (por ejemplo, una ventana de 60 días), se cambia.

**Diseño de las grillas (por qué no se prueban valores que difieren en una unidad, como 29 y 30 días):** la diferencia de error entre valores tan cercanos sería menor que la variación entre semillas (±2 %, notebook 06e), así que ninguna medición podría distinguirlos. Por eso los puntos de cada grilla están espaciados (por ejemplo 7, 14, 30, 60 y 90 días). Se aplica a todas las grillas: ventana, ancla y embedding, cabezas de atención de B y ventana de detección de saltos.

| Prueba | Qué fundamenta | Tiempo |
|---|---|---|
| **Hecha:** etiquetado fuera de servicio con tramos de 3, 5, 7 y 14 días | R5 (7 y 5 días): insensible | — |
| **Hecha (exploratoria):** interpolación con huecos de 1 a 14 días y relación voltaje–corriente | R3 y R4: definen las variantes a probar | — |
| **Hecha:** paro (10 o 13,5 kA) × interpolación (3 o 12 días), con la CV de 3 bloques, 3 semillas y los mismos objetivos en todas las variantes. Resultado: **10 kA y 12 días** (sección 6.2) | R3 y R4 | 40 min |
| Umbral (0,05, 0,10, 0,15 V) × ventana (3, 5, 7, 14 días) de detección de saltos, con el mismo procedimiento (afecta la máscara de saltos del entrenamiento) | R9 | ~1 h |
| Ventana de entrada de 7, 14, 30, 60 y 90 días (validación cruzada de 3 bloques, 3 semillas, menor error). El VMD causal se calcula una sola vez con 90 días y las ventanas más cortas se toman de ahí | L | ~1 h |
| Grilla ancla {3, 7, 14} × embedding {4, 8, 16} (menor error) | Bi-LSTM | ~40 min |
| Mediciones de lote y separación de orígenes (Bi-LSTM, B, C) | Lote, orígenes cada 5 y cada 10 días | ~20 min |
| Cabezas de atención {1, 2, 4, 8} (opcional) | B | ~1 h |
| ETS con α y β continuos (opcional) | ETS | minutos |

---

## 9. Orden de trabajo (rama `feature/test`)

El orden importa porque cada paso cambia los datos que usa el siguiente: las pruebas que usan la señal filtrada (ventana, ancla, saltos) tienen que hacerse con la señal VMD definitiva, es decir, después de aplicar el BO-VMD de Makri.

1. **Pruebas de limpieza** (R3, R4 y R5): no dependen del VMD.
2. **Aplicar la limpieza y el BO-VMD de Makri**, y reconstruir Silver y Gold (notebooks 04, 05 y 05b).
3. **Pruebas que usan la señal filtrada:** ventana de entrada, ventana de detección de saltos (R9), ancla y embedding, mediciones de lote y, si se hace, cabezas de atención de B.
4. **Cambios de configuración y de código de los modelos:** módulo espacial de Palan para C, Kalman de Khan para B, la configuración de la sección 4, y quitar la v2 de la comparación (notebook 07, ranking, README).
5. **Reentrenar todos los modelos** (notebooks 06, 06b, 06c y 06d).
6. **Ejecutar de nuevo el notebook 07, una sola vez**, y actualizar el README, `docs/metricas.md` y `docs/intentos_mejora.md`.
7. Merge a `main`.

**Tiempo estimado:** se actualiza al cerrar el inventario de la sección 10 (las pruebas de limpieza con el criterio de menor error suman ~2 h).

---

## 10. Inventario de valores que no estaban mapeados

Surgió de una revisión completa de `config.py`, de los módulos de `src/voltcast/` y de los números escritos a mano en los notebooks. Cada fila indica dónde está el valor, por qué hoy no tiene justificación y cómo se propone fundamentarlo.

### 10.1 Datos y evaluación (afectan a todos los modelos)

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| Orígenes de evaluación cada 7 días (`EVAL_STRIDE`) | `config.py`; validación, bloque interno, CV y ETS | **Hecho (A).** En los bloques de train, el RMSE de persistencia y tendencia con orígenes cada 7 días difiere entre −2,1 % y +1,8 % del obtenido con orígenes diarios | **Justificado:** todos los modelos se evalúan en los mismos orígenes, así que el espaciado no favorece a ninguno; separar los orígenes una semana reduce la correlación entre ellos (lo que necesita el bootstrap de C2), y el VMD causal cuesta 7 veces menos (~21 000 frente a ~146 000 descomposiciones). Las métricas absolutas tienen una incertidumbre de ese orden (±2 %) |
| Cortes de 60 y 180 días (`EVAL_HORIZONS`) | `config.py` | Vienen de la especificación | **F:** son los extremos del rango de 60–180 días que pide la especificación (`docs/prompt_v1.md`); 180 días está justificado en la sección 6.3 |
| 181 celdas físicas por electrolizador (`N_CELDAS_FISICAS`) | `config.py` | Dato de planta | **A:** el notebook 03 (sección 6) muestra que `[Volts]` corresponde a 181 celdas (≈ 3,4 V por cada una de las 31 que no están en el Excel). Hay que corregir el Benchmarking, que dice 183 |
| Recorte de covariables a \|z\| ≤ 5 (`COV_CLIP`) | `config.py`, Gold | Elegido sin medición | **Prueba:** contar cuántos valores superan \|z\| = 5. Si son pocos o ninguno, el valor no influye (A). Si no, decidirlo por menor error |
| Regla de "día informativo" para fuera de servicio (kA normal y dato en la mayoría de las celdas) | `cleaning.out_of_service_mask` | **Hecho (A).** Con dato en más de 1/4, 1/2 o 3/4 de las celdas, los días-celda válidos de train son idénticos (553 953; 0 de diferencia) | **Justificado: insensible** |
| Con duplicados, conservar el último registro del día | `cleaning.daily_wide` (R2) | **Hecho (A).** Solo 4 días tienen más de un registro (3 en el EL A, 1 en el EL B); entre el primero y el último, el voltaje cambia en promedio 0,0007 y 0,0023 V, menos que el cambio diario medio (0,0026 V) | **Justificado: insensible** |
| Reentrenar con todo train tras el early stopping (`REFIT_FULL_TRAIN`) | `config.py`, notebooks 06 y 06b–06d | Elegido sin medición | **Prueba** de menor error en la CV de 3 bloques: con y sin reentrenamiento |
| Semillas: 1 por modelo (42) | `config.py` | El valor de la semilla no necesita justificación; la cantidad sí | **F:** las fuentes entrenan un solo modelo. Se informa la variación entre semillas medida en el notebook 06e (±2 %) |

### 10.2 Modelo propio (Bi-LSTM y Bi-NOA-LSTM)

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| Objetivo relativo al ancla y escalado por σ_Δ | `dataset.build_inputs` | Contribución propia; el respaldo es indirecto (el notebook 07, sección 5.1, muestra el sesgo de nivel de los modelos con voltaje absoluto) | **Prueba:** ablación en la CV de 3 bloques (relativo al ancla frente a voltaje absoluto) |
| Entradas: kA, V_total / celdas en servicio, canal de máscara | `dataset.FEATURES` | kA y V_total tienen respaldo cualitativo (notebook 03, sección 6); el canal de máscara no tiene ninguno | **Prueba:** ablación en la CV (con y sin cada entrada) |
| Indicador de electrolizador en el cabezal | `model.VoltCastNet` | Sin justificación | **Prueba:** ablación en la CV |
| Activación GELU del cabezal | `model.VoltCastNet` | Sin fuente | **F:** usar ReLU, como Song (capa densa con ReLU) y Gou |
| Optimizador AdamW | `model.fit` (todos los modelos) | Las fuentes usan Adam | **F:** usar Adam, con la regularización L2 como `weight_decay` (equivale a la L2 de Khan) |
| Pérdida MSE enmascarada | `model.masked_mse` | — | **F:** MSE como en Makri y Khan; la máscara excluye objetivos sin dato válido |

### 10.3 Propuesta A (BO-VMD, RFR-SHAP)

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| BO-VMD sobre una muestra de 20 celdas estratificada (`BO_SAMPLE_CELLS`) y regla para usar un par común de K y α (`choose_params`: diferencia de K ≤ 1, razón de α ≤ 2) | `config.py`, `vmd.py`, notebook 05 | Elegidos sin fuente | **F:** Makri aplica BO-VMD a **cada serie que pronostica**. Aplicarlo a cada una de las 300 celdas (~50 min en paralelo) elimina la muestra y la regla de homogeneidad |
| Parámetros internos del VMD: τ = 0, DC = 0, init = 1, tolerancia 1e-7 | `vmd.py` | Sin cita | **F / pendiente de cita:** la documentación de `vmdpy` recomienda τ = 0 ("para tolerar ruido") e indica una tolerancia "típicamente alrededor de 1e-6". Usar 1e-6 o citar el ejemplo oficial de la biblioteca |
| Random Forest de RFR-SHAP: 300 árboles, profundidad 8, mínimo 20 por hoja, 2000 muestras para SHAP | `features.rfr_shap` | Makri publica 100 árboles (sec. 3.3) y no los demás parámetros | **Aplicado. F:** 100 árboles (Makri); los demás, los valores por defecto de scikit-learn, que es la biblioteca que usa Makri (A) |
| Candidatas y variable objetivo de RFR-SHAP | notebook 06c | Makri ordena cada variable por separado (corrientes, voltajes, modos) y elige las 12 más importantes para predecir la serie | **A:** candidatas = los 15 tipos de modo de la celda, kA, el voltaje medio por celda y el calendario (seno y coseno del día del año y de la semana), 21 en total, con las celdas juntas; objetivo = el voltaje medido del día siguiente a partir de las variables del día (el análogo del pronóstico). La entrada del MIMO son los modos elegidos de cada celda y las covariables elegidas |
| Filas del Random Forest (14 544 pares celda-día al azar) | notebook 06c | Con los valores por defecto los árboles crecen sin límite y el costo de SHAP crece con las hojas: con ~64 000 filas, 100 muestras no terminaron en 10 minutos (medido) | **A (medición):** el tamaño del dataset de Makri (101 días cada 10 minutos, sec. 3.1) |
| Muestras para SHAP (1000) | `features.rfr_shap` | Makri no lo publica | **A (medición):** el notebook 06c verifica que el top-12 es el mismo con 250 y con 500 muestras; si cambiara, se informa |

### 10.4 Propuesta B (Kalman y Seq2Seq)

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| Varianza inicial del filtro (P0 = 1e-2 y 1e-6) y 10 observaciones de calentamiento (`BURN_IN`) | `kalman.py` | Sin fuente | **A:** reemplazar por la inicialización difusa estándar de los modelos de espacio de estados (libro de referencia, pendiente de cita), o probar que el resultado no cambia |
| Punto de partida de la máxima verosimilitud (`x0`) | `kalman.fit_mle` | Solo es el punto inicial del optimizador | **Prueba (en el notebook 05b):** la estimación conjunta se repite desde tres puntos de partida y se usa la de mayor verosimilitud; el notebook informa la diferencia |
| Reiniciar el filtro en saltos y arranques al estimar | `kalman.py`, notebook 05b | Sin fuente | **Prueba (en el notebook 05b):** estimación con y sin reinicios. Los saltos se detectan con el umbral de diagnóstico de R9 (0,10 V), porque el reinicio, como el diagnóstico, busca separar intervenciones |
| Observaciones del Kalman | notebook 05b | — | **A:** solo voltajes medidos (sin los interpolados de Silver): el filtro predice sin corregir en los días sin dato, que es su forma estándar de tratar huecos. La estimación usa las 300 celdas (ya no hay muestra BO) |
| Decoder de B, detalle por detalle | `model.Seq2SeqAttention` | Gou et al. (2024) pronostica **un solo paso** (la hora siguiente), así que no describe un decoder que genere 180 días | Ver la tabla 10.4.1 |

#### 10.4.1 Decoder de B

| Detalle | Implementación actual | Fuente | Estado |
|---|---|---|---|
| Atención multi-cabeza (producto escalado y proyección de salida) | Sí | Gou, ecuaciones 12 y 13 | **F** |
| Capa de salida sobre [estado del decoder, contexto de atención] | 2 capas con GELU | Gou, ecuación 14: una sola capa totalmente conectada | **F**, pasando a una sola capa |
| Generación paso a paso de los 180 días | Sí | Benchmarking, Propuesta B ("decodificación paso a paso") | **F** |
| Decoder unidireccional (LSTM) | Sí | Gou usa un decoder BiLSTM porque recorre la historia ya observada de cada región para dar un solo valor; un decoder que genera el futuro paso a paso no puede recorrer hacia atrás valores que todavía no generó | **A** (la bidireccionalidad se mantiene en el encoder, que recorre la historia, como en Gou) |
| Estado inicial del decoder desde el encoder, con una capa lineal y tanh | Sí | Cho et al. (2014), EMNLP, apéndice A.1.1: h′₀ = tanh(V′c); Sutskever, Vinyals y Le (2014), NeurIPS: el estado inicial del decoder es el último estado del encoder | **F** (fuente externa al Benchmarking) |
| Embedding de celda e indicador de electrolizador en el estado inicial | Sí | — | **A:** da a B las mismas condiciones que a nuestro modelo, que comparte pesos entre las 300 celdas |
| Entrada de cada paso: el valor del paso anterior | Sí | Sutskever et al. (2014), ec. 1: cada salida se condiciona en las salidas anteriores | **F** (fuente externa) |
| Entrada del primer paso: el último voltaje observado (relativo al ancla) | Sí | Sutskever usa un símbolo de inicio, porque trabaja con palabras | **A:** en una serie numérica, el análogo es el último valor observado |
| Consulta (query) de la atención: el estado del decoder del paso anterior | Sí | Bahdanau, Cho y Bengio (2015), ICLR, ec. 6: eᵢⱼ = a(sᵢ₋₁, hⱼ) | **F** (fuente externa) |
| Entrenamiento con las propias predicciones (sin *teacher forcing*) | Sí | En Sutskever y Cho, el entrenamiento se condiciona en los valores reales anteriores (*teacher forcing*) | **Prueba** de menor error (con y sin *teacher forcing*), junto con la de cabezas de atención |
| Número de cabezas de atención | 4 | No lo publica ninguna fuente | **Prueba** {1, 2, 4, 8} |

**Sobre las fuentes externas:** Sutskever et al. (NeurIPS 2014, >17 000 citas en OpenAlex), Cho et al. (EMNLP 2014, >25 000) y Bahdanau et al. (ICLR 2015, >16 000) son los artículos que definieron la arquitectura codificador-decodificador y el mecanismo de atención; los tres se verificaron en su texto completo (arXiv 1409.3215, 1406.1078 y 1409.0473). Tratan de traducción automática, así que se citan **solo para convenciones de arquitectura**, no para afirmar nada sobre el rendimiento en pronóstico.

### 10.5 Propuesta D (TCN)

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| Convolución inicial de suavizado con kernel 5 | `model.TCNMultiOutput` | Sin fuente | **Aplicado. F:** se elimina; ni Duan ni Palan la usan |
| Bloque residual con 2 convoluciones causales y ReLU | `model.TCNBlock` | Verificado: Duan (sec. de métodos, Fig. 7c) dice que "a residual block in TCN consists of two layers of dilated causal convolutions" y cita a Lea et al. (2017, CVPR; su ref. 36) | **F** (Duan) |
| Convolución 1×1 en la conexión residual del primer bloque | `model.TCNBlock` | Al quitar la convolución de suavizado, el primer bloque pasa de 151 canales (celdas + kA) a 64 y la suma residual necesita igualar los canales | **F (fuente externa):** Bai, Kolter y Koltun (2018), el TCN genérico (arXiv 1803.01271), usan una convolución 1×1 cuando cambian los canales; Duan lo cita (su ref. 37) |

### 10.6 Evaluación y criterios

| Valor | Dónde | Situación | Cómo fundamentarlo |
|---|---|---|---|
| Bootstrap de 1000 iteraciones con intervalo del 95 % (C2) | `metrics.bootstrap_rmse_diff` | — | **F:** Khan usa intervalos del 95 % estimados con bootstrap de 1000 iteraciones |
| C3: RMSE < 0,15 V en al menos el **95 %** de las celdas | `docs/metricas.md`, notebook 07 | El 0,15 V viene de la Tabla 10 del OB2; el 95 % no tiene fuente | **Sin fundamento.** C3 es un criterio propio y complementario: o se justifica el 95 % o se declara como límite |
| C4: R² sobre el cambio > 0 | `docs/metricas.md` | — | **A:** es el mismo umbral que el R² > 0 de la Tabla 10, aplicado al cambio de voltaje |
| Celdas críticas de los gráficos (mayor voltaje y mayor error) | notebook 07 | Solo para presentar resultados | No afecta métricas ni decisiones; no requiere fundamentación |

---

## 11. Mejoras posibles (no se ejecutan en esta PoC)

Quedan documentadas como trabajo futuro. Ninguna es necesaria para fundamentar la configuración actual, pero cada una eliminaría una limitación declarada.

| # | Mejora | Qué resuelve | Método (fuente) | Tiempo estimado |
|---|---|---|---|---|
| 1 | Búsqueda amplia de hiperparámetros de la Bi-LSTM propia | La configuración se fundamenta con valores de fuentes y grillas chicas; una búsqueda podría mejorarla | Optuna (TPE), hasta 100 trials con corte tras 20 sin mejora (Khan), con el espacio de la sección 7 | ~6 h |
| 2 | SOMA de la Propuesta A con los parámetros de Makri | A usa el óptimo que Makri obtuvo con sus datos, no con los nuestros | SOMA (población 6, *path length* 4, paso 0,11, espacio de 8 hiperparámetros), cortando cuando el líder no mejora en una migración completa (Makri, sec. 2.4.3, paso 6) | 10–20 h por variante (A1 y A2) |
| 3 | Búsqueda de hiperparámetros de la Propuesta B | Ídem, con el óptimo de Khan | ATPE/TPE con el espacio de Khan (Tabla 10), incluidos q, R y α del Kalman; 100 trials con corte tras 20 sin mejora | ~8 h |
| 4 | Búsqueda de hiperparámetros de la Propuesta C | Ídem, con el óptimo de Palan | Optuna con el espacio de Palan (Tabla 3), 20 trials | ~2–3 h |
| 5 | Grilla de la Propuesta D | D usa la configuración fija de Duan y Palan | Grid search de Morcillo (Tabla 2), 24 combinaciones | ~0,5 h |
| 6 | **Prueba de ventana de entrada para cada modelo del Benchmarking** | Hoy los modelos del Benchmarking usan 30 días (especificación, igual para todos) y solo la Bi-LSTM propia usa la ventana elegida por su prueba (sección 6.4). Ninguna fuente publica una ventana en días que se pueda trasladar | Grilla de 7, 14, 30, 60 y 90 días con el criterio de menor error, para cada modelo | ~1 h por modelo (~5 h en total) |

Las mejoras 1 a 5 suman las **40–55 h de GPU** del camino 2 (sección 3); con la 6, unas 45–60 h.
