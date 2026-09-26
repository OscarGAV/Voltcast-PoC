# DETALLE ARQUITECTÓNICO Y APLICACIÓN DE PROPUESTAS DE MACHINE LEARNING / DEEP LEARNING PARA EL PRONÓSTICO DE VOLTAJE POR CELDA

## 1. INTRODUCCIÓN Y CONTEXTO DEL DATASET INDUSTRIAL

El presente documento detalla la formulación matemática, la arquitectura algorítmica, los mecanismos de filtrado de señal, las técnicas de optimización y el procedimiento de aplicación práctica de las cinco (5) propuestas de modelos predictivos evaluadas en el Benchmarking.

### Estructura del Dataset de Entrada
El conjunto de datos de la planta química se organiza como una matriz temporal donde:
*   **Filas (Dimensión Temporal $T$):** Registros horativos continuos ($t_1, t_2, \dots, t_N$).
*   **Columnas (Variables y Celdas $N$):** Voltajes individuales de las $183$ celdas de alta tensión ($V_1, V_2, \dots, V_{183}$) distribuidas en la Sección A y Sección B, junto con covariables operativas globales como la intensidad de corriente en kiloamperios ($kA$) y variables de calendario (hora del día, día de la semana).
*   **Objetivo de Pronóstico:** Transformar una ventana histórica de entrada de $L$ horas/días (por ejemplo: $L = 720 \text{ horas}$ o $30 \text{ días}$) en una matriz proyectada de salida de $H$ horas/días futuros ($H = 1440 \text{ a } 4320 \text{ horas}$ o $60 \text{ a } 180 \text{ días}$) para todas las celdas de forma simultánea o secuencial.

---

## 2. DETALLE TÉCNICO Y APLICACIÓN DE CADA PROPUESTA

---

### PROPUESTA A: BO-VMD + SOMA-Bi-NOA-LSTM (MIMO)
*Referencias Principales: Makri et al. (2026); analisis-modelos-secuenciales-02.md; analisis-modelos-secuenciales-03.md; Accurate electricity consumption forecasting for industrial energy management.pdf*

#### 1. Arquitectura y Componentes Algorítmicos
*   **Técnica de Filtrado y Descomposición:** **BO-VMD (Bayesian Optimization - Variational Mode Decomposition)**. Descompone señales no estacionarias y volátiles en $K$ Funciones de Modo Intríseco (IMFs) adaptativas con frecuencias delimitadas, aislando el ruido electromagnético de alta frecuencia del piso de planta.
*   **Selección Explicable de Características:** **RFR-SHAP (Random Forest Regressor + SHapley Additive exPlanations)**. Identifica la importancia física no lineal de las covariables (ejemplo: impacto de la corriente $kA$ sobre cada celda).
*   **Optimización de Hiperparámetros:** **SOMA (Self-Organizing Migrating Algorithm)**. Algoritmo heurístico estocástico basado en comportamiento de enjambre para encontrar la tasa de aprendizaje, cantidad de capas y unidades ocultas óptimas.
*   **Red Neuronal Recurrente:** **Bi-NOA-LSTM (Bidirectional Long Short-Term Memory without Output Activation)**. Red bidireccional que procesa secuencias en sentido hacia adelante y hacia atrás. La modificación **NOA (Non-Output Activation)** elimina la función de activación no lineal (sigmoide/tanh) de la compuerta de salida, evitando la saturación de gradientes durante picos bruscos de voltaje o corriente.
*   **Paradigma de Salida:** **MIMO (Multi-Input Multi-Output)**. Mapeo directo matriz-a-matriz.

#### 2. Metodología de Aplicación a Nuestro Dataset
1.  **Transformación Tensor de Entrada:** Se extrae una ventana deslizante histórica de $L = 720 \text{ horas}$ ($30 \text{ días}$). El tensor de entrada adopta la forma $[B, L, C]$, donde $B$ es el tamaño de lote, $L=720$ pasos temporales y $C = 184$ (183 celdas + 1 columna de $kA$).
2.  **Filtrado Adaptativo por Celda (BO-VMD):** Cada serie temporal horaria de voltaje $V_i(t)$ se pasa por BO-VMD. Los modos de alta frecuencia (ruido) se descartan o atenúan, reconstruyendo la señal limpia $V_i^{\text{filtrado}}(t)$.
3.  **Procesamiento Bi-NOA-LSTM:** El tensor limpio pasa a través de las capas Bi-NOA-LSTM. El procesamiento bidireccional captura dependencias pasadas y patrones de retorno operativo.
4.  **Inferencia MIMO en un solo Pase (*Single Forward Pass*):** La capa densa de proyección final multiplica internamente las representaciones ocultas y emite directamente el tensor de salida $[B, H, 183]$, proyectando los voltajes de las 183 celdas para las $H = 4320 \text{ horas}$ ($180 \text{ días}$) futuras de golpe.
5.  **Desnormalización:** Se aplica la transformación inversa Min-Max para recuperar los voltajes absolutos reales en voltios ($V$).

---

### PROPUESTA B: Seq2Seq BiLSTM con Mecanismo de Atención y Filtrado de Kalman
*Referencias Principales: Khan et al. (2026); Song et al. (2025); Deep neuro-stochastic ensemble for lighting energy consumption forecasting.pdf*

#### 1. Arquitectura y Componentes Algorítmicos
*   **Técnica de Filtrado:** **Filtro de Kalman Estocástico**. Modelo de estimación de estados recursivo que filtra el ruido gaussiano midiendo la varianza de la telemetría.
*   **Mecanismo de Atención:** **Multi-Head Self-Attention**. Asigna ponderaciones dinámicas a diferentes pasos de tiempo históricos para resaltar momentos de carga máxima o cambios de turno.
*   **Red Neuronal:** **Seq2Seq BiLSTM (Encoder-Decoder)**. Un *Encoder* BiLSTM comprime la secuencia de entrada en un vector de estado latente, y un *Decoder* LSTM genera las salidas futuras.
*   **Paradigma de Salida:** **Decodificación Secuencial Autorregresiva Paso a Paso**.

#### 2. Metodología de Aplicación a Nuestro Dataset
1.  **Preprocesamiento:** Se aplica el Filtro de Kalman sobre las filas horarias de las 183 celdas para suavizar la curva de voltaje.
2.  **Codificación (Encoder):** La ventana de $L=720 \text{ horas}$ se introduce al *Encoder* BiLSTM, produciendo representaciones vectoriales latentes.
3.  **Cálculo de Pesos de Atención:** El mecanismo de atención calcula la matriz de alineamiento entre los estados del *Encoder* y del *Decoder*.
4.  **Decodificación Iterativa:** El *Decoder* genera el voltaje de las 183 celdas para el paso $t+1$. Para calcular $t+2$, re-inyecta la predicción de $t+1$ como entrada de forma recursiva hasta completar las $H$ horas futuras.

---

### PROPUESTA C: ST-GRU (Spatial-Temporal Gated Recurrent Unit)
*Referencias Principales: Palan & Sumith N. (2026); Research on deep learning construction of energy user profiles considering environmental awareness.pdf*

#### 1. Arquitectura y Componentes Algorítmicos
*   **Componente Espacial:** **Spatial GRU / Graph Convolutional Network (GCN)**. Modela las dependencias físicas y el acoplamiento eléctrico entre celdas vecinas (Sección A y B) utilizando una matriz de adyacencia $\mathbf{A}$.
*   **Componente Temporal:** **Temporal GRU**. Celdas Recurrentes de Gating que capturan la evolución temporal de cada celda con menor consumo computacional que LSTM.
*   **Técnica de Filtrado:** Normalización espacial y convolución de grafos sobre la topología de la planta.
*   **Paradigma de Salida:** **Mapeo Espaciotemporal Coordinado**.

#### 2. Metodología de Aplicación a Nuestro Dataset
1.  **Construcción del Grafo de Celdas:** Se define un grafo donde las 183 celdas son nodos y las aristas representan conexiones físicas/eléctricas o correlaciones de voltaje entre ellas.
2.  **Extracción Espacial (Spatial GRU):** En cada hora $t$, la capa espacial procesa la fila de 183 celdas y aprende cómo la caída de voltaje en la celda $i$ afecta a la celda $j$.
3.  **Extracción Temporal (Temporal GRU):** Las características espaciales se procesan a través de la GRU temporal sobre la secuencia de $L$ horas.
4.  **Generación de Pronóstico:** Se proyecta la evolución futura de todo el grafo de celdas sobre el horizonte $H$.

---

### PROPUESTA D: TCN Multi-Output (Temporal Convolutional Networks)
*Referencias Principales: Morcillo-Jimenez et al. (2024); High-precision short-term industrial energy consumption forecasting via parallel-NN.pdf*

#### 1. Arquitectura y Componentes Algorítmicos
*   **Estructura Convolucional:** **Convoluciones Causales Dilatadas 1D (Dilated Causal Convolutions)**. Convoluciones que respetan la causalidad temporal (no miran al futuro) con un factor de dilatación $d = 1, 2, 4, 8 \dots$ que expande el campo receptivo exponencialmente sin aumentar desmedidamente los parámetros.
*   **Conexiones Residuales:** **Residual Blocks**. Bloques con salto de conexión (*skip-connections*) que previenen la degradación del entrenamiento en redes profundas.
*   **Técnica de Filtrado:** Convoluciones de suavizado integradas en las primeras capas.
*   **Paradigma de Salida:** **Proyección Densa Multi-Output Paralela**.

#### 2. Metodología de Aplicación a Nuestro Dataset
1.  **Formateo de Entrada:** La matriz de $[L, 184]$ se trata como un tensor de canales de audio/señal 1D de 184 canales a lo largo de $L$ pasos temporales.
2.  **Paso Convolucional Dilatado:** Se aplican múltiples capas TCN con dilatación creciente para cubrir las 720 horas históricas.
3.  **Capa Densa Final Multi-Salida:** Un cabezal lineal denso proyecta simultáneamente la matriz final $[H, 183]$ en una sola operación convolucional/lineal paralela.

---

### PROPUESTA E: Informer / PatchTST (Transformers de Largo Alcance)
*Referencias Principales: Guan Min (2026); Zhang et al. (2026); Long-term sequence forecasting in industrial grids using PatchTST.pdf*

#### 1. Arquitectura y Componentes Algorítmicos
*   **Mecanismo de Atención Probabilística / Divisoria:**
    *   **ProbSparse Self-Attention (Informer):** Selecciona los vectores *Key* dominantes para reducir la complejidad computacional de $O(L^2)$ a $O(L \log L)$.
    *   **Patching (PatchTST):** Divide las series de tiempo horarias en sub-secuencias o "parches" (ejemplo: ventanas de 24 horas) tratándolos como tokens.
*   **Independencia de Canales (Channel Independence):** Cada celda se procesa como una serie temporal independiente compartiendo los pesos del Transformer.
*   **Paradigma de Salida:** **Generación Directa con Decodificador Generativo Distilado**.

#### 2. Metodología de Aplicación a Nuestro Dataset
1.  **Parcheado de la Serie Horaria:** Las 720 horas históricas de cada celda se agrupan en parches de 24 horas, creando $30$ tokens por celda.
2.  **Incrustación Espaciotemporal (Embedding):** Se agregan codificaciones de posición temporal y de identificación de celda.
3.  **Atención ProbSparse / Patch Transformer:** Las capas del Encoder procesan la secuencia de parches identificando tendencias semanales y mensuales.
4.  **Proyección Generativa:** El Decoder genera la matriz proyectada para las $H$ horas futuras.

---

## 3. TABLA RESUMEN COMPARATIVA DE APLICACIÓN PRÁCTICA

| Aspecto | Propuesta A (BO-VMD + Bi-NOA-LSTM) | Propuesta B (Seq2Seq + Atención) | Propuesta C (ST-GRU Espacial) | Propuesta D (TCN Multi-Output) | Propuesta E (PatchTST / Informer) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arquitectura Base** | Bi-NOA-LSTM (MIMO) | BiLSTM Encoder-Decoder | Spatial-Temporal GRU | Convolución Causal Dilatada (TCN) | Transformer basado en Parches |
| **Filtrado de Ruido** | BO-VMD (Descomposición Adaptativa) | Filtro de Kalman | Convolución sobre Grafo | Convolución 1D Suavizadora | Normalización por Parche |
| **Manejo de Celdas ($183$)** | Matriz paralela con $kA$ covariable | Canales individuales con Atención | Nodos en Grafo de Adyacencia $\mathbf{A}$ | Canales de Entrada Convolucionales | Independencia de Canales (*Channel Ind.*) |
| **Generación Futura ($H$)** | Single Forward Pass (1 solo cálculo) | Iterativo Paso a Paso ($t+1 \dots t+H$) | Proyección Progresiva Espaciotemporal | Capa Densa Paralela | Decodificador Generativo por Parches |
| **Riesgo de Acumulación de Error** | **Nulo (0%)** | Alto (*Error Drift*) | Moderado | Nulo (0%) | Bajo |
| **Latencia de Inferencia** | **8 – 12 ms** | 50 – 150 ms | 15 – 30 ms | < 8 ms | > 200 ms |


## 4. CONSIDERACIONES

- No esta 100% asegurado que todo lo que indique el MD sea correcto

- Si identificas oportunidades de mejorar, debes consultarmelas y explicarmelas primero

- Todo se hará en notebooks

- Te recomiendo primero analizar la estructura del dataset de excel