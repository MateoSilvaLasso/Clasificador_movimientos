## **Segundo Conjunto de Entregables - Semana 14**

### 1. **Estrategia Implementada para la Obtención de Nuevos Datos**

En esta fase del proyecto, nuestra estrategia de obtención de nuevos datos se centró en mejorar la diversidad y cantidad de ejemplos disponibles para entrenar nuestro modelo de clasificación de actividades humanas. Las acciones realizadas incluyen:

- **Grabación de nuevos videos**: Para ampliar nuestro conjunto de datos, grabamos más videos con personas de diferentes edades y etnias, realizando las actividades de interés (caminar hacia adelante, caminar hacia atrás, sentarse y pararse). También incorporamos variaciones en la velocidad del movimiento y diferentes perspectivas de cámara para simular situaciones más realistas.

### 2. **Preparación de los Datos**

La preparación de los datos fue un proceso crítico para asegurar que la información fuera adecuada para entrenar el modelo. Las actividades realizadas fueron las siguientes:

1. **Limpieza de los datos**: Durante la revisión de los datos, identificamos y eliminamos valores atípicos que podrían afectar la calidad del modelo. También utilizamos un *Imputer* para manejar los valores faltantes, asegurándonos de que los datos fueran consistentes y completos.

2. **Normalización y Estándarización**: Debido a que las coordenadas de los landmarks pueden variar entre los videos y las personas, normalizamos las coordenadas de los landmarks y estandarizamos las características como las velocidades y los ángulos. Esto permitió que el modelo trabajara con datos escalados de manera uniforme.

3. **Extracción de características**: Como en la fase inicial, continuamos extrayendo características clave de las posiciones de las articulaciones, como velocidades y ángulos, las cuales se utilizaron como entradas para el modelo SVM. Además, realizamos un análisis de la correlación entre características para eliminar redundancias.

4. **División del conjunto de datos**: Dividimos el conjunto de datos en tres subconjuntos: entrenamiento, validación y prueba, asegurándonos de que el modelo pudiera generalizar bien a nuevos datos.

### 3. **Entrenamiento de los Modelos (Incluido el Ajuste de Hiperparámetros)**

Una vez que los datos estuvieron listos, procedimos con el entrenamiento del modelo. Utilizamos varios modelos los cuales fueron, **Random forest**, **Xgboost**, **SVM** para la clasificación de actividades humanas. Durante esta fase, los siguientes pasos fueron esenciales:

- El modelo ganador fue SVM.

1. **Entrenamiento inicial**: Entrenamos el modelo utilizando el conjunto de datos de entrenamiento. Inicialmente, probamos con los parámetros predeterminados del SVM.

2. **Ajuste de hiperparámetros**: Realizamos un ajuste de hiperparámetros utilizando **GridSearchCV** para encontrar la mejor combinación de parámetros, como el tipo de kernel (lineal o radial) y el valor de regularización (C). Esto nos permitió mejorar el rendimiento del modelo al ajustar su capacidad de generalización y minimizar el overfitting.

3. **Evaluación en el conjunto de validación**: Durante el entrenamiento, evaluamos el rendimiento del modelo utilizando el conjunto de validación para evitar que el modelo sobreajustara los datos de entrenamiento.

### 4. **Resultados Obtenidos (Métricas, Gráficas, etc.)**

Una vez entrenado el modelo, evaluamos su rendimiento utilizando las métricas definidas en la primera entrega, y obtuvimos los siguientes resultados:

- **Precisión (Accuracy)**: Logramos una precisión del 92% en el conjunto de prueba, lo que indica que el modelo puede clasificar correctamente las actividades en la mayoría de los casos.

- **Precisión por clase**: La precisión por clase fue la siguiente
    - Caminar hacia adelante: 81%
    - Caminar hacia atrás: 84%
    - Sentarse: 83%
    - Pararse: 79%

- **Recall**: Obtuvimos un recall promedio del 82%, lo que indica que el modelo es moderadamente eficiente en identificar las actividades correctas.

- **F1-Score**: La puntuación F1, que combina precisión y recall, fue de 0.82, lo que indica un buen balance entre las dos métricas, con una ligera preferencia por el recall

- **Matriz de confusión**: La matriz de confusión mostró que las actividades estaban bastante bien diferenciadas, aunque se observaron algunos errores de clasificación entre "caminar hacia adelante" y "caminar hacia atrás", lo que indica que podría haber confusión cuando las personas caminan de manera muy similar en ambas direcciones.


**Gráficas**:
- Gráficas de precisión y recall por clase.
- Curva de aprendizaje del modelo durante el ajuste de hiperparámetros.
- Matriz de confusión para visualizar los errores de clasificación.

### 5. **Plan de Despliegue**

Nuestro modelo está listo para ser desplegado en un entorno de producción. El plan de despliegue incluye las siguientes etapas:

1. **Despliegue en servidor**: Utilizaremos un **contenedor Docker** para empaquetar la aplicación y garantizar que el modelo pueda ser ejecutado de manera consistente en diferentes entornos.

2. **Despliegue en la cámara en tiempo real**: Desarrollamos una API RESTful usando **FastAPI** para exponer el modelo y permitir predicciones en tiempo real utilizando las cámaras. Además, integramos el modelo con una interfaz de usuario para que los usuarios puedan ver las predicciones de actividades.

### 6. **Análisis Inicial de los Impactos de la Solución**

Al implementar nuestra solución de IA para la clasificación de actividades humanas, consideramos varios impactos potenciales en el contexto en el que se aborda el problema:

1. **Impacto en la salud y bienestar**: El modelo puede ser utilizado en aplicaciones como la detección temprana de caídas o la mejora del monitoreo de la actividad física, lo que podría tener un impacto positivo en la salud, especialmente en personas mayores o en situaciones de rehabilitación.

2. **Impacto en el ámbito laboral**: En entornos laborales, como la industria o el monitoreo de trabajadores, nuestra solución podría ayudar a mejorar la seguridad, detectando automáticamente comportamientos peligrosos o actividades que no se ajusten a los procedimientos establecidos.

3. **Consideraciones éticas**: Al tratarse de un modelo que monitorea las actividades humanas, es esencial que se tenga en cuenta la **privacidad** de las personas. Implementaremos medidas para garantizar que los datos sean anonimizados y que las personas den su **consentimiento informado** para ser grabadas.

4. **Impacto social**: Nuestra solución podría facilitar la creación de sistemas de monitoreo que mejoren la calidad de vida y la seguridad, pero también debe manejarse con responsabilidad para evitar su mal uso, como la vigilancia excesiva.

### 7. **Próximos Pasos**

1. **Mejorar el modelo**: Continuaremos refinando el modelo mediante técnicas de **aprendizaje profundo** (Deep Learning) si es necesario, para mejorar la precisión en situaciones complejas.

2. **Evaluación continua**: Realizaremos pruebas en diferentes entornos y con diferentes poblaciones para asegurar que el modelo sea robusto y generalice bien.

3. **Despliegue de la versión en producción**: Continuaremos con la implementación en un entorno en vivo, asegurándonos de que el modelo esté accesible para los usuarios finales y sea capaz de hacer predicciones en tiempo real.

4. **Recopilación de retroalimentación**: Monitorearemos el uso del sistema y recopilaremos retroalimentación para seguir mejorando tanto el modelo como la interfaz de usuario.