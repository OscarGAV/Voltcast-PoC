# Principales Historias de Usuario - Sistema VoltCast

Este documento recopila el listado resumido de las **Historias de Usuario** del sistema VoltCast

---

## Índice de Módulos

1. [Módulo de Ingesta, Preprocesamiento y ETL (US001 - US004)](#1-módulo-de-ingesta-preprocesamiento-y-etl)
2. [Módulo de Machine Learning, Pronósticos y Evaluación (US005 - US008)](#2-módulo-de-machine-learning-pronósticos-y-evaluación)
3. [Módulo de Gestión de Celdas y Componentes Eléctricos (US009)](#3-módulo-de-gestión-de-celdas-y-componentes-eléctricos)

---

## 1. Módulo de Ingesta, Preprocesamiento y ETL

### US001: Carga de Archivos Históricos

* **Código:** `US001`
* **RF Asociado:** `RF01`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero subir archivos en formato CSV o Excel con registros históricos de voltaje, corriente y factor de potencia, para tener un modelo entrenado con los patrones de la empresa.
* **Criterios de Aceptación:**
  * **E01 (Carga exitosa):** **Dado que** el Ingeniero de Planta está autenticado en el sistema **Y** cuenta con un archivo CSV o Excel con registros de voltaje, **Cuando** el ingeniero sube el archivo al sistema, **Entonces** el sistema muestra la confirmación del registro exitoso.
  * **E02 (Formato inválido):** **Dado que** el Ingeniero de Planta está autenticado en el sistema **Y** el archivo seleccionado tiene una extensión no soportada, **Cuando** el ingeniero intenta subir el archivo, **Entonces** el sistema rechaza la carga **Y** muestra un mensaje de error especificando la causa del rechazo.
  * **E03 (Estructura inválida):** **Dado que** el Ingeniero de Planta está autenticado en el sistema **Y** el archivo seleccionado no cuenta con alguna columna que exprese valores de voltaje, **Cuando** el ingeniero intenta subir el archivo, **Entonces** el sistema rechaza la carga **Y** muestra un mensaje de error especificando la causa del rechazo.

---

### US002: Identificación de Columnas

* **Código:** `US002`
* **RF Asociado:** `RF02`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero contar con un sistema que identifique el tipo de columnas que tiene un archivo Excel, para determinar rápidamente las columnas con valores de voltaje y corriente eléctrica.
* **Criterios de Aceptación:**
  * **E01 (Identificación correcta):** **Dado que** el Ingeniero de Planta carga un archivo Excel en el sistema **Y** el archivo cuenta con encabezados reconocibles de voltaje, corriente y factor de potencia, **Cuando** el sistema procesa el archivo, **Entonces** el sistema identifica y clasifica correctamente cada columna según su tipo.
  * **E02 (Columnas no identificables):** **Dado que** el Ingeniero de Planta carga un archivo Excel en el sistema **Y** el archivo no cuenta con encabezados reconocibles, **Cuando** el sistema intenta identificar el tipo de columnas, **Entonces** el sistema muestra un mensaje indicando que no pudo identificar columnas **Y** solicita al usuario verificar la estructura del archivo.

---

### US003: Identificación de Celdas Eléctricas en Archivo

* **Código:** `US003`
* **RF Asociado:** `RF03`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero contar con un sistema que identifique las celdas eléctricas de un archivo Excel, para tener un modelo adaptado a las condiciones actuales de la planta.
* **Criterios de Aceptación:**
  * **E01 (Identificación exitosa):** **Dado que** el Ingeniero de Planta carga un archivo Excel con registros de celdas eléctricas, **Cuando** el sistema procesa el archivo, **Entonces** el sistema identifica correctamente cada celda eléctrica registrada, indicando el número de celdas eléctricas.
  * **E02 (Sin celdas reconocibles):** **Dado que** el Ingeniero de Planta ha cargado un archivo Excel que no contiene registros identificables como celdas eléctricas, **Cuando** el sistema intenta identificar las celdas, **Entonces** el sistema muestra un mensaje indicando que no se encontraron celdas eléctricas.

---

### US004: Limpieza y Ordenamiento de Registros

* **Código:** `US004`
* **RF Asociado:** `RF04`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero contar con un sistema que ordene los registros de un archivo Excel, para evitar valores duplicados y datos no relacionados con el voltaje.
* **Criterios de Aceptación:**
  * **E01 (Limpieza exitosa):** **Dado que** el Ingeniero de Planta carga un archivo Excel **Y** el sistema identifica correctamente las columnas y celdas eléctricas, **Cuando** el sistema procesa los registros del archivo, **Entonces** el sistema ordena los registros y elimina los datos duplicados o no relacionados con el voltaje.
  * **E02 (Detección de duplicados):** **Dado que** el archivo contiene registros duplicados, **Cuando** el sistema realiza el proceso de limpieza, **Entonces** el sistema conserva únicamente un registro por cada valor duplicado **Y** descarta los datos no relacionados con el voltaje.

---

## 2. Módulo de Machine Learning, Pronósticos y Evaluación

### US005: Generación de Proyecciones de Voltaje

* **Código:** `US005`
* **RF Asociado:** `RF05`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero probar el modelo predictivo cuando esté entrenado, para generar proyecciones de voltaje.
* **Criterios de Aceptación:**
  * **E01 (Generación exitosa):** **Dado que** el modelo predictivo se encuentra disponible y entrenado con los patrones de la empresa, **Cuando** el Ingeniero de Planta solicita generar una proyección de voltaje, **Entonces** el sistema genera y muestra las proyecciones de voltaje correspondientes.
  * **E02 (Modelo en reentrenamiento):** **Dado que** el modelo predictivo está en proceso de reentrenamiento, **Cuando** el Ingeniero de Planta intenta generar una proyección, **Entonces** el sistema muestra un mensaje indicando que se usará la última versión disponible del modelo y que una nueva versión saldrá próximamente.

---

### US006: Evaluación de la Precisión del Modelo

* **Código:** `US006`
* **RF Asociado:** `RF06`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero poder evaluar la precisión del modelo, para conocer la efectividad del sistema.
* **Criterios de Aceptación:**
  * **E01 (Cálculo exitoso):** **Dado que** el modelo predictivo cuenta con proyecciones generadas y datos reales de comparación, **Cuando** el Ingeniero de Planta solicita evaluar la precisión del modelo, **Entonces** el sistema calcula y muestra las métricas de precisión (MSE, RMSE, MAE, MAPE).
  * **E02 (Modelo en reentrenamiento):** **Dado que** el modelo predictivo está en proceso de reentrenamiento, **Cuando** el Ingeniero de Planta solicita evaluar la precisión del modelo, **Entonces** el sistema muestra un mensaje indicando que se usará la última versión disponible del modelo.

---

### US007: Visualización de Pronósticos de Voltaje

* **Código:** `US007`
* **RF Asociado:** `RF07`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero visualizar pronósticos de voltaje generados, para identificar celdas que superen el voltaje permitido.
* **Criterios de Aceptación:**
  * **E01 (Visualización general):** **Dado que** el sistema cuenta con pronósticos de voltaje generados, **Cuando** el Ingeniero de Planta accede a la sección de pronósticos, **Entonces** el sistema muestra los pronósticos de voltaje de todas las celdas.
  * **E02 (Resaltado de excesos):** **Dado que** existen pronósticos que superan el umbral de voltaje permitido, **Cuando** el Ingeniero de Planta visualiza los pronósticos, **Entonces** el sistema resalta visualmente las celdas cuyo pronóstico excede el voltaje permitido.
  * **E03 (Consulta offline exitosa):** **Dado que** el sistema cuenta con pronósticos previamente consultados, **Cuando** el Ingeniero de Planta accede a la sección de pronósticos sin conexión a Internet, **Entonces** el sistema muestra los pronósticos previamente consultados.
  * **E04 (Consulta offline fallida):** **Dado que** el Ingeniero de Planta no cuenta con pronósticos previamente consultados, **Cuando** accede a la sección sin conexión a Internet, **Entonces** el sistema informa que no existen pronósticos disponibles para consulta sin conexión.

---

### US008: Filtrado de Pronósticos por Fecha

* **Código:** `US008`
* **RF Asociado:** `RF08`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero filtrar pronósticos de voltajes por fecha, para visualizar pronósticos de una fecha específica.
* **Criterios de Aceptación:**
  * **E01 (Filtrado por fecha exitoso):** **Dado que** el sistema cuenta con pronósticos de voltaje generados para distintas fechas, **Cuando** el Ingeniero de Planta selecciona una fecha específica, **Entonces** el sistema muestra únicamente los pronósticos correspondientes a esa fecha.
  * **E02 (Filtrado exitoso sin conexión):** **Dado que** el sistema cuenta con pronósticos almacenados localmente para distintas fechas, **Cuando** el ingeniero no cuenta con conexión a Internet, **Entonces** el sistema muestra los pronósticos generados disponibles localmente.
  * **E03 (Filtrado fallido sin conexión):** **Dado que** el sistema no cuenta con pronósticos almacenados localmente para la fecha seleccionada **Y** no hay conexión a Internet, **Entonces** el sistema informa que no existen datos disponibles para consulta sin conexión.

---

## 3. Módulo de Gestión de Celdas y Componentes Eléctricos

### US009: Registro de Celdas Eléctricas

* **Código:** `US009`
* **RF Asociado:** `RF09`
* **Rol:** Ingeniero de Planta
* **Historia:** Como Ingeniero de Planta, quiero registrar las celdas eléctricas de la empresa, para tener la información de las celdas en el sistema.
* **Criterios de Aceptación:**
  * **E01 (Registro exitoso):** **Dado que** el Ingeniero de Planta ingresa los datos requeridos de una nueva celda eléctrica, **Cuando** confirma el registro, **Entonces** el sistema almacena la celda eléctrica **Y** muestra un mensaje de confirmación.
  * **E02 (Datos incompletos):** **Dado que** el Ingeniero de Planta no ha completado todos los campos obligatorios, **Cuando** intenta registrar la celda, **Entonces** el sistema rechaza el registro **Y** muestra un mensaje indicando los campos faltantes.
