### Entrega 3: Semana 17  

#### Reducción de Características  
En esta etapa, implementamos una reducción de características para optimizar el modelo, mejorar su rendimiento y facilitar la interpretación de los resultados. Para ello:  
1. **Análisis de Importancia de Características**: Utilizamos técnicas como `SelectFromModel` con un clasificador basado en árboles para identificar las características más relevantes.  
2. **Eliminación de Características Redundantes**: Removimos variables altamente correlacionadas que no aportaban significativamente al desempeño del modelo.  
3. **Resultados de la Reducción**: No pudimos reducir las caracteristicas sin afectar la eficiencia del modelo por lo tanto no aplicamos reduccion de caracteristicas.  

#### Evaluación de Resultados  
Tras la reducción de características, realizamos una nueva evaluación del modelo:  
- **Precisión Promedio**: 0.69%, tuvo una disminución bastante significativa.
- **Recall Promedio**: 0,71%, manteniendo un buen nivel de detección de actividades.  
- **Tiempo de Predicción**: pero aunque el modelo bajaba pudimos notar que la prediccion en tiempo real se hacia de forma mas rapida. 

#### Despliegue de la Solución  
Desplegamos la solución final en un entorno accesible para el cliente, con las siguientes características:  

2. **Backend**: Creamos un pkl con el modelo entrenado donde se pueden hacer consultas sobre el modelo.  
3. **Frontend**: Diseñado con Streamlit, ofreciendo una interfaz intuitiva para el usuario final.  
4. **Documentación**: Entregamos un manual de usuario y una guía técnica para facilitar la adopción de la solución(aclaraciones.md).  

#### Entrega al Cliente  s
La solución fue entregada al cliente como un paquete listo para su implementación, incluyendo:  
- El modelo entrenado y optimizado.  
- Código fuente organizado y documentado.  
- Instrucciones para instalación, uso y solución de problemas comunes.  

#### Análisis Final de Impacto  
La solución tiene impactos positivos en varios niveles:  
- **Social**: Facilita el monitoreo de actividades físicas, promoviendo el bienestar personal.  
- **Económico**: Ofrece una herramienta accesible y económica en comparación con soluciones comerciales similares.  
- **Ético**: Respetamos la privacidad de los datos al no almacenar información personal y procesar todo localmente.  

#### Video de Presentación  
Incluimos un video de 10 minutos que presenta:  
1. El contexto del problema.  
2. Las técnicas utilizadas a lo largo del proyecto.  
3. Los resultados obtenidos en cada etapa.  
4. Los logros alcanzados, incluyendo la precisión del modelo y su impacto.  

#### Conclusión  
Con esta entrega final, consolidamos nuestro proyecto como una solución práctica, ética y efectiva para la clasificación de actividades físicas en tiempo real.