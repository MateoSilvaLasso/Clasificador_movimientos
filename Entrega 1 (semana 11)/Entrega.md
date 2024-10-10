## **Primer Conjunto de Entregables - Semana 11**

### 1. **Pregunta(s) de interés**

En nuestro proyecto, la pregunta principal es:

- **¿Es posible clasificar diferentes actividades humanas (caminar hacia adelante, caminar hacia atrás, sentarse, pararse, girar) utilizando datos procesados de las posiciones y movimientos de las articulaciones humanas?**

Nuestro objetivo es utilizar las coordenadas de las articulaciones obtenidas mediante MediaPipe para predecir distintas actividades humanas. Esto implica analizar cómo se mueve el cuerpo humano y cómo estas características pueden ser utilizadas para identificar con precisión la actividad que está realizando una persona.

### 2. **Tipo de problema**

Este es un problema de **clasificación multiclase**, ya que nuestro objetivo es predecir una de varias categorías de actividad. Las actividades que intentamos clasificar son:

- Caminar hacia adelante
- Caminar hacia atrás
- Sentarse
- Pararse
- girar

Usamos un **modelo de Support Vector Machine (SVM)** para resolver este problema de clasificación. Este tipo de modelo es adecuado para problemas con un número relativamente pequeño de muestras y cuando las clases pueden no ser linealmente separables.

### 3. **Metodología**

Nuestra metodología se basa en los siguientes pasos:

1. **Recolección de Datos**: Recopilamos datos a partir de videos donde personas realizan actividades como caminar, sentarse y pararse. Estos datos fueron procesados con la herramienta MediaPipe para extraer las posiciones de las articulaciones de los sujetos en cada frame de video.

2. **Preprocesamiento de Datos**: Tras extraer los datos de los landmarks, aplicamos técnicas de preprocesamiento para extraer características adicionales, como velocidades de movimiento y ángulos de articulaciones clave (codos, tronco). Además, normalizamos las coordenadas para que el modelo pueda trabajar con datos escalados y estandarizados.

3. **Entrenamiento del Modelo**: Entrenamos un modelo SVM utilizando las características extraídas de los datos de los videos. El modelo es capaz de aprender a identificar las actividades humanas basándose en los patrones de movimiento y las posiciones de las articulaciones.

4. **Evaluación**: Evaluamos el rendimiento del modelo mediante métricas estándar como precisión, recall, F1-score, y matriz de confusión para asegurar que el modelo pueda clasificar correctamente las actividades.

5. **Mejoras**: En función de los resultados obtenidos, planeamos realizar ajustes en los hiperparámetros del modelo y continuar refinando el proceso de extracción de características si es necesario.

### 4. **Métricas para medir el progreso**

Para evaluar el rendimiento de nuestro modelo y medir el progreso, utilizamos las siguientes métricas de clasificación:

- **Exactitud (Accuracy)**: Proporción de predicciones correctas respecto al total de predicciones.
- **Precisión (Precision)**: Porcentaje de predicciones positivas correctas.
- **Recall (Sensibilidad)**: Proporción de verdaderos positivos identificados sobre el total de positivos reales.
- **F1-Score**: Promedio ponderado de la precisión y el recall, útil cuando tenemos clases desbalanceadas.
- **Matriz de Confusión**: Nos ayuda a ver en qué clases el modelo está cometiendo más errores, permitiéndonos identificar áreas de mejora.

### 5. **Datos recolectados**

Para este proyecto, hemos recolectado datos a través de videos que muestran personas realizando diferentes actividades. Los datos procesados de MediaPipe incluyen las posiciones de las articulaciones (x, y, z) de cada frame, además de la visibilidad de cada landmark. Estos datos se almacenan en archivos CSV, y cada archivo contiene información sobre los siguientes campos:

- **frame**: El número del frame en el video.
- **landmark_index**: El índice de cada articulación (por ejemplo, codo, hombro, rodilla).
- **x, y, z**: Las coordenadas de cada landmark.
- **visibility**: Un valor que indica la visibilidad de la articulación en ese frame.

Además, hemos extraído características adicionales como las **velocidades de las articulaciones** y **ángulos entre las articulaciones clave** (por ejemplo, codo derecho, codo izquierdo, tronco) que se utilizan como entradas para el modelo SVM.

### 6. **Análisis exploratorio de los datos (EDA)**

Durante nuestro análisis exploratorio de los datos, nos enfocamos en lo siguiente:

- **Distribución de las clases**: Observamos que las clases de actividades están balanceadas, pero realizamos una verificación adicional para asegurarnos de que no haya un sesgo en los datos. Si encontramos que una clase tiene menos muestras, podríamos aplicar técnicas de balanceo como el *oversampling*.
  
- **Visualización de las características**: Creamos gráficos para visualizar la distribución de las coordenadas de los landmarks, las velocidades y los ángulos de las articulaciones. Esto nos ayudó a detectar patrones que podrían estar relacionados con las actividades que estamos clasificando.

- **Correlación entre características**: Analizamos la correlación entre las distintas características (por ejemplo, la velocidad de los codos y el tronco) para verificar si algunas de ellas son redundantes y podrían eliminarse para simplificar el modelo.

- **Detección de outliers**: Identificamos algunos valores atípicos que podrían estar afectando el rendimiento del modelo. Decidimos aplicar una estrategia de imputación para corregir estos outliers.

### 7. **Estrategias para conseguir más datos**

Sabemos que, para mejorar la precisión y robustez del modelo, es crucial contar con un conjunto de datos lo suficientemente diverso. Por eso, hemos propuesto las siguientes estrategias para conseguir más datos:

1. **Grabación de más videos**: Si es necesario, grabaremos más videos con diferentes personas realizando las actividades de interés, incluyendo variaciones como caminar más rápido, diferentes ángulos de visión, o variaciones en cómo las personas se sientan.

2. **Uso de datasets públicos**: Existen datasets como **Human3.6M** o **UCF101** que contienen videos con diversas actividades humanas etiquetadas. Podemos utilizar estos conjuntos de datos para complementar nuestro propio conjunto.

3. **Aumentación de datos**: Aplicaremos técnicas de aumentación de datos para simular variaciones, como rotación, cambio de escala, y adición de ruido a los datos existentes, lo que permitirá generar más ejemplos sin la necesidad de grabar nuevos videos.

### 8. **Aspectos éticos al implementar soluciones de IA**

Al implementar una solución basada en IA para la clasificación de actividades humanas, hemos identificado varios aspectos éticos que debemos considerar:

- **Privacidad**: Dado que estamos utilizando datos de videos de personas, es esencial que obtengamos el consentimiento informado de los participantes y que sus datos sean anonimizados. Debemos asegurarnos de que la información personal no sea divulgada sin autorización.

- **Bias (sesgo)**: Nos aseguramos de que los datos sean representativos de diversas poblaciones (por ejemplo, género, edad, raza) para evitar que el modelo aprenda sesgos que puedan afectar la precisión de las predicciones para ciertos grupos.

- **Transparencia**: El modelo debe ser explicable, especialmente si se va a utilizar en contextos sensibles, como la salud o la seguridad. Nos aseguramos de que los usuarios puedan entender cómo se toman las decisiones.

- **Impacto social**: Reflexionamos sobre el impacto potencial de la automatización de la clasificación de actividades humanas, especialmente en ámbitos laborales o de monitoreo, y nos comprometemos a usar la tecnología de manera responsable.

### 9. **Próximos pasos**

A continuación, nuestros siguientes pasos son:

1. **Ajustar los hiperparámetros** del modelo SVM para mejorar el rendimiento, usando técnicas como GridSearchCV para encontrar la mejor combinación de parámetros.
2. **Recolección de más datos** para mejorar la precisión y robustez del modelo.

3. **Entrenar otros modelos**: Crear nuevos modelos va a ser escencial para probar cual se puede ajustar mejor a nuestra solucion.

4. **Implementación de un sistema de predicción en tiempo real**, utilizando la cámara para realizar predicciones sobre las actividades sin necesidad de duplicar el código de preprocesamiento.