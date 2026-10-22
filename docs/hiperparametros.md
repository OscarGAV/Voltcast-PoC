# Fundamentación de configuraciones e hiperparámetros

Objetivo: que **cada valor** de configuración de la PoC tenga un fundamento verificable y no sea una suposición.

Cada valor se clasifica así:

| Marca | Significado |
|---|---|
| **F** | Tomado de una fuente del Benchmarking (se indica artículo y tabla o sección). |
| **A** | Adaptación de una fuente a nuestros datos o recursos, con el motivo explícito y verificable. |
| **P** | Criterio propio sin fuente. **Pendiente**: debe reemplazarse por un valor de fuente o por una búsqueda de hiperparámetros. |

Datos de la PoC que motivan las adaptaciones: datos **diarios** (las fuentes usan datos de 5 min a 1 h), 300 celdas, ~1 945 días de train, horizonte de 1 a 180 días y una GPU RTX 4060 de 8 GB.

---

## 1. Fuentes revisadas

PDF en `TP202620029/Fuentes - Benchmarking` y artículos de acceso abierto.

| Prop. | Fuente | Método de búsqueda | Presupuesto | Espacio de búsqueda | Óptimo publicado |
|---|---|---|---|---|---|
| A | Makri et al. (2026), *Results in Engineering* 29 | SOMA (compara con Hyperband y PSO) | Población 6, 100 migraciones, *path length* 4, paso 0,11 (Tabla 5) | Capas {1, 2, 3}; unidades {20, 40, …, 180}; dropout {0–0,5}; LR {1e-4, 1e-3, 3e-3, 1e-2}; épocas {8, 16, 32, 64}; lote {50, 100}; pérdida {MSE, MAE}; optimizador {Adam, Adadelta} (Tabla 1) | 3 capas, 51 unidades, dropout 0, LR 3e-3, 150 épocas, lote 16 (Tabla 6). **Inconsistencia:** unidades, épocas y lote quedan fuera de su propio espacio |
| A | Makri (BO-VMD) | Optimización bayesiana con proceso gaussiano (scikit-optimize) | 30 evaluaciones | K ∈ [2, 15], α ∈ [400, 2000]; objetivo: error de reconstrucción (ec. 14) | K = 15, α = 500 |
| A | Makri (RFR-SHAP) | — | — | — | Se seleccionan las 12 variables más importantes (umbral fijo) |
| B | Khan et al. (2026), *Energy Reports* 15 | ATPE (TPE adaptativo); en la Tabla 9 lo comparan con grilla y búsqueda aleatoria y gana | **100 trials por modelo; corta tras 20 trials sin mejora** | LR [1e-5, 1e-2] logarítmico; lote {16, 32, 64, 128}; dropout [0,1; 0,5]; BiLSTM: capa 1 [128, 512], capa 2 [64, 256]; capas densas [128, 512] y [64, 256]; Kalman q y R [1e-6, 1e-2] logarítmicos; α de fusión [0,3; 0,7] (Tabla 10) | BiLSTM (200, 100), densas (256, 128), secuencia de 24, lote 32, Adam 1e-3, dropout 0,2, L2 1e-4, 200 épocas con paciencia 20, CV de 5 particiones (Tabla 7); α* = 0,58–0,62 (Tabla 8); Kalman con estado [nivel, velocidad, aceleración] |
| B | Song et al. (2025), *Energy and AI* 20 | Optuna (TPE) | No indicado | No publicado para la red | CNN de 128 filtros, 2 BiLSTM de 128, dropout 0,3, L2 0,01, LR 1e-4, lote 64, paciencia 20, máximo 100 épocas (sec. 3.2) |
| C | Palan & N (2026), *Scientific Reports* 16 | Optuna | 20 trials | Unidades {64, 128, 256, 512}; capas {1–4}; LR [2e-4, 5e-3] logarítmico; épocas 70–130 (Tabla 3) | 512 unidades, 2 capas, dropout 0,2 (fijo), LR 7,69e-4, 90 épocas; Adam con weight decay y early stopping. **"Espacial" se refiere a relaciones entre variables, no a un grafo físico** |
| D | Morcillo-Jimenez et al. (2024), *Applied Intelligence* 54 | Grid search | 24 combinaciones | Capas {1, 2}; dropout {0,15; 0,3; 0,5}; neuronas {16, 32, 64, 128} (Tabla 2) | Seq2Seq: 2 capas, 64 neuronas, dropout 0,3 (Tabla 3) |
| D | Duan et al. (2026), *npj Adv. Manufacturing* 3 | Ninguno (configuración fija) | — | — | TCN de 3 capas convolucionales (32, 64 y 128 filtros), kernel 3, LR 1e-3, lote 32 |
| D | Palan & N (2026), TCN de comparación | Fija | — | — | 64 canales, kernel 3, 4 capas, dropout 0,2 (Tabla 4) |
| E | Guan Min (2026), *Energy Informatics* 9 | Algoritmo genético | 50 generaciones × 30 individuos | Filtros {16, 32, 64}; kernel {3, 5, 7}; unidades BiLSTM {32, 64, 128}; LR {1e-4, 1e-3, 1e-2} | — |
| E | Zhang et al. (2026), *ECM: X* 31 | Pruebas preliminares; "no se hizo búsqueda automática" | — | — | LSTM 2 × 128, dropout 0,1, LR 1e-3, lote 256, 100 épocas |

**Conclusiones de la revisión:**
- 6 de las 8 fuentes usan una búsqueda automática; ninguna prueba todas las combinaciones.
- Khan (Tabla 9) muestra que, con el mismo presupuesto, la búsqueda tipo TPE supera a la grilla y a la búsqueda aleatoria. Optuna usa TPE por defecto, igual que Song y Palan.

---

## 2. Modelo propio (Bi-LSTM): búsqueda de hiperparámetros

La configuración actual (`config.TRAIN`) es **P**: se fijó con valores estándar sin una búsqueda. Se reemplaza por una búsqueda fundamentada.

- **Método:** Optuna con TPE. Fuentes: Song, Palan y Khan (Tabla 9, TPE frente a grilla y búsqueda aleatoria).
- **Presupuesto:** hasta 100 trials, con corte tras 20 trials sin mejora (Khan, sec. 3.7).
- **Validación:** validación cruzada temporal de 3 bloques dentro de train (notebook 06e), sin usar el período de validación.
  - **A:** Khan usa CV de 5 particiones, pero en series de tiempo las particiones deben respetar el orden temporal.
- **Objetivo:** promedio entre bloques del RMSE relativo; el R² sobre el cambio es criterio secundario.
- **Validación final:** la configuración ganadora se evalúa una sola vez en la validación (notebook 07).

### Espacio de búsqueda y su fuente

| Hiperparámetro | Rango | Escala | Marca | Fundamento |
|---|---|---|---|---|
| Learning rate | [1e-4, 1e-2] | logarítmica | F | Makri (Tabla 1) y Guan; incluye los óptimos de Palan (7,7e-4) y Khan (1e-3) |
| Unidades de la LSTM | [32, 256] | logarítmica | A | Makri {20–180}, Khan capa 2 [64, 256], Guan {32–128}. Se excluyen los 512 de Palan y de la capa 1 de Khan: con ~1 945 días por celda, un modelo mayor sobreajusta (riesgo 3 del Project Charter) |
| Capas | {1, 2, 3} | entera | F | Makri (Tabla 1); el óptimo de Palan es 2 |
| Capa densa de salida | [128, 512] | logarítmica | F | Khan, capa densa 1 (Tabla 10) |
| Dropout | [0, 0,5] | lineal | F | Makri (Tabla 1) y Morcillo; Khan [0,1; 0,5] |
| Regularización (weight decay) | [1e-6, 1e-3] | logarítmica | A | Khan y Song usan L2 (1e-4 y 1e-2). El rango se centra en el óptimo de Khan |
| Lote | {256, 512, 1024, 2048} | categórica | A | Las fuentes usan 16–128 con miles de muestras; nosotros tenemos ~108 000 por época, y un lote de 16 haría cada época ~60 veces más lenta. Zhang usa 256 |
| Épocas | early stopping con paciencia 20, máximo 200 | — | F | Khan (200 / 20) y Song (100 / 20) |
| Ventana de entrada | {30, 60, 90} días | categórica | F / A | 30 días es lo que pide la especificación (`docs/prompt_v1.md`, L = 720 h). Guan y Palan también optimizan la ventana. Hasta 90 días para no perder demasiados orígenes de train |
| Ancla (días promediados) | [3, 14] | entera | P → búsqueda | No está en ninguna fuente porque es una contribución propia. Se fundamenta con la búsqueda; el rango va de casi el último valor a dos semanas, siempre menor que la ventana |
| Embedding de celda | [4, 16] | entera | P → búsqueda | No está en las fuentes (contribución propia: pesos compartidos entre celdas). Se fundamenta con la búsqueda |

Otros elementos de la configuración:
- **Máscara de saltos:** **A**, fundamentada por la ablación del notebook 06.
- **Celda LSTM frente a NOA:** **A**, por la ablación de los notebooks 06 y 07.
- **Ensamble de 5 semillas:** **A**, por la validación cruzada del notebook 06e.

---

## 3. Propuesta A (BO-VMD + RFR-SHAP + SOMA + Bi-NOA-LSTM MIMO)

| Elemento | PoC | Makri (2026) | Marca | Observación |
|---|---|---|---|---|
| Arquitectura MIMO, Bi-NOA-LSTM y Min-Max | igual | igual | F | — |
| BO-VMD: evaluaciones | 30 | 30 | F | — |
| BO-VMD: algoritmo | TPE (Optuna) | Proceso gaussiano (skopt) | P | Debe usarse el proceso gaussiano, como en la fuente |
| BO-VMD: rango de K | [3, 10] | [2, 15] | P | Debe ser el de la fuente |
| BO-VMD: rango de α | [500, 5000] | [400, 2000] | P | Debe ser el de la fuente |
| BO-VMD: objetivo | Mínima entropía de envolvente | Error de reconstrucción (ec. 14) | P | El objetivo de Makri baja siempre al aumentar K, lo que explica su K = 15 en el máximo del rango. Usar el de la fuente o justificar el cambio con otra fuente |
| RFR-SHAP: selección | Importancia ≥ 5 % | Las 12 variables más importantes | A | Nuestro dataset tiene solo 4 covariables candidatas (kA, voltaje medio de celda, día del año y día de la semana), así que "top-12" las incluiría a todas. Con el umbral del 5 % se seleccionaron kA y el día del año |
| SOMA: parámetros | Población 5, 3 migraciones, *path length* 2, paso 0,4, PRT 0,3 | Población 6, 100 migraciones, *path length* 4, paso 0,11 | P | Ver la sección 3.1 |
| SOMA: hiperparámetros buscados | LR, capas, unidades (3) | Capas, unidades, dropout, LR, épocas, lote, pérdida y optimizador (8) | P | Ver la sección 3.1 |
| SOMA: rangos | LR [1e-4, 1e-2], capas [1, 3], unidades [32, 256] | Tabla 1 | P | A2 terminó en los bordes (LR = 0,01 y 256 unidades): el espacio quedó corto |
| Dropout del entrenamiento | 0,2 | 0 (Tablas 3 y 6) | P | Debe ser el de la fuente o entrar en la búsqueda |
| Weight decay | 1e-4 | No usa | P | — |
| Lote | 32 | 32 (base, Tabla 3) | F | — |
| Épocas | Early stopping (máximo 80, paciencia 10) | Fijas: 50 en la base, 150 en el óptimo | P | — |

### 3.1 SOMA: por qué no se usó la configuración original

**Costo real de la configuración de Makri:**
- En SOMA AllToOne, cada migración mueve a 5 individuos (población 6 menos el líder) durante *path length* ÷ paso = 4 ÷ 0,11 ≈ 36 pasos. Son unas **180 evaluaciones por migración**, y con 100 migraciones, **unas 18 000 evaluaciones**.
- En nuestra PoC cada evaluación es entrenar el MIMO, unos 1,2 minutos, porque A1 tardó 73 min en 61 evaluaciones. **Eso da unos 15 días por variante, unos 30 días para A1 y A2.**
- Makri trabaja con un pronóstico a 1 hora y datos cada 10 min, así que cada evaluación suya es mucho más barata.

**Lo que se hizo fue reducir el presupuesto sin una fuente que respalde los valores elegidos.** Por eso hoy es **P**. La sección 7 detalla cómo se corrige.

---

## 4. Propuesta B (Seq2Seq BiLSTM + atención + Kalman)

| Elemento | PoC | Fuente | Marca | Observación |
|---|---|---|---|---|
| Learning rate | 1e-3 | Khan: Adam 1e-3 (Tabla 7) | F | — |
| Dropout | 0,2 | Khan: 0,2 (Tabla 7) | F | — |
| Regularización | weight decay 1e-4 | Khan: L2 1e-4 | F | — |
| Unidades | encoder 64 (bidireccional), decoder 128 | Khan: BiLSTM (200, 100); Song: 2 × 128 | P | Deben venir de la fuente o de su búsqueda (ATPE en el rango de la Tabla 10) |
| Cabezas de atención | 4 | No figura en Khan ni en Song (la atención multi-cabeza viene de Gou et al. 2024, citado por Khan) | P | — |
| Lote | 512 | Khan 32; Song 64 | A | La memoria de la GPU (8 GB) con un decoder de 180 pasos y 108 000 muestras. Falta medir el límite real |
| Épocas / paciencia | 40 / 6 | Khan 200 / 20; Song 100 / 20 | P | Se redujo por costo; la paciencia de 20 es de la fuente |
| Orígenes cada 10 días | 10 | — | A | Cada época con el decoder de 180 pasos es cara. Falta fundamentarlo |
| Ensamble de semillas | 3 | Khan no lo usa | P | — |
| α de fusión: rango | [0,30; 0,70] | Khan [0,3; 0,7] (Tabla 10) | F | El óptimo quedó en 0,70, en el borde del rango |
| Kalman: estado | nivel y pendiente | Khan: nivel, velocidad y aceleración (descripción del filtro de Kalman, junto a la Tabla 6) | P | Debe ser el de la fuente o justificar el cambio |
| Kalman: q y R | Máxima verosimilitud | Khan: ATPE en [1e-6, 1e-2] | A | La máxima verosimilitud es el método estándar para modelos de espacio de estados; falta citar la referencia |

---

## 5. Propuesta C (ST-GRU)

| Elemento | PoC | Palan & N (2026) | Marca | Observación |
|---|---|---|---|---|
| Dropout | 0,2 | 0,2 (Tabla 3) | F | — |
| Unidades de la GRU / capas | 64 / 1 | 512 / 2 (óptimo de Optuna) | P | Usar el método de la fuente (Optuna con su espacio de la Tabla 3) o su óptimo |
| Learning rate | 1e-3 | 7,69e-4 | P | — |
| Weight decay | 1e-4 | Usa Adam con weight decay, sin publicar el valor | P | — |
| Componente espacial | GCN sobre un grafo de celdas (2 vecinas en serie + 4 de mayor correlación) | Relaciones **entre variables** de entrada, sin grafo físico | P | Es nuestra interpretación del texto del Benchmarking ("interacciones físicas entre variables") |
| Lote / épocas | 16 / early stopping (80, 10) | Épocas 70–130, con early stopping | P | — |

---

## 6. Propuesta D (TCN Multi-Output)

| Elemento | PoC | Fuente | Marca | Observación |
|---|---|---|---|---|
| Kernel | 3 | Duan; Palan (Tabla 4) | F | — |
| Canales | 64 | Palan, 64 canales (Tabla 4) | F | Duan usa 32/64/128 |
| Dropout | 0,2 | Palan (Tabla 4) | F | — |
| Learning rate | 1e-3 | Duan | F | — |
| Lote | 32 | Duan | F | — |
| Bloques y dilataciones | 5 (1, 2, 4, 8, 16) | Palan: 4 capas; Duan: 3 | A | Se eligió para que el campo receptivo cubra la ventana de 30 días. Con kernel 3, 4 capas ya cubren 61 días, así que la de Palan alcanzaría |
| Weight decay | 1e-4 | No figura | P | — |
| Épocas | Early stopping (80, 10) | Duan: 20 épocas + 50 de ajuste fino | P | — |
| Búsqueda | Ninguna | Morcillo: grid search (Tabla 2) | P | Aplicar la grilla de Morcillo, adaptada a TCN (neuronas → canales) |

---

## 7. Plan para que todo quede fundamentado

Cada propuesta se configura con **el método de búsqueda y el espacio de su propia fuente**, sobre nuestros datos. Así ninguna queda con una configuración elegida a mano, y la comparación es justa: cada modelo recibe el ajuste que su autor considera apropiado.

| Modelo | Método (fuente) | Presupuesto (fuente) | Costo estimado (RTX 4060) |
|---|---|---|---|
| Bi-LSTM propia | Optuna TPE (Song, Palan, Khan) | 100 trials, corte tras 20 sin mejora (Khan) | ~6 h |
| A1 / A2 | SOMA con los parámetros de Makri (población 6, *path length* 4, paso 0,11) y su espacio (Tabla 1) | Ver la decisión pendiente | Ver la decisión pendiente |
| A: BO-VMD | Proceso gaussiano, K [2, 15], α [400, 2000], 30 evaluaciones (Makri) | 30 evaluaciones (Makri) | ~2 min |
| B | ATPE/TPE con el espacio de Khan (Tabla 10), incluidos q, R y α | 100 trials, corte tras 20 sin mejora (Khan) | ~8 h |
| C | Optuna con el espacio de Palan (Tabla 3) | 20 trials (Palan) | ~2–3 h |
| D | Grid search de Morcillo (Tabla 2) | 24 combinaciones (Morcillo) | ~0,5 h |

### Decisión pendiente: SOMA de la Propuesta A

Con los parámetros de Makri, una migración cuesta unas 180 evaluaciones (~3,6 h por variante). Opciones:

1. **Parámetros de Makri con un criterio de parada de su propio algoritmo.** Makri (sec. 2.4.3, paso 6) admite terminar por "número máximo de migraciones o umbral de convergencia". Se corta cuando el líder no mejora en una migración completa, con un tope de tiempo. Todos los parámetros vienen de la fuente; lo único que cambia es la parada, que la fuente permite. Unas 10–20 h por variante.
2. **Configuración completa de Makri (100 migraciones).** Unos 15 días por variante. Inviable dentro del cronograma del proyecto.
3. **Mantener el SOMA reducido actual** y declarar la reducción como límite de cómputo. Es la opción más débil, porque los valores reducidos no tienen fuente.

**Recomendación: opción 1.**
