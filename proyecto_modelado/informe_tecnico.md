# Informe Técnico: Modelado Predictivo para Detección de Retrasos en Envíos

## 1. Resumen Ejecutivo
El presente proyecto tiene como objetivo desarrollar un sistema predictivo capaz de anticipar si un envío sufrirá un retraso en su entrega (`es_retraso` = 1) o llegará a tiempo (`es_retraso` = 0). Utilizando el framework de Kedro para estructurar y ejecutar los pipelines de ingestión, limpieza e integración de datos, se consolidó una **Tabla Maestra** (`master_table.parquet`). Sobre esta base, se diseñaron, entrenaron y evaluaron modelos de Machine Learning (Regresión Logística y Random Forest) que logran identificar de forma confiable los retrasos futuros, optimizando así la cadena logística y permitiendo acciones preventivas.

## 2. Marco Metodológico
Para abordar la problemática como una tarea de **clasificación binaria**, se optó por dos algoritmos fundamentales:
- **Regresión Logística**: Elegido por su interpretabilidad y simplicidad como modelo base (baseline). Permite entender la relación lineal entre características (como distancia o costos de incidencias) y la probabilidad de retraso.
- **Random Forest Classifier**: Un algoritmo de ensamble robusto frente a relaciones no lineales e interacciones complejas entre variables. Fue seleccionado como modelo avanzado debido a su capacidad para manejar múltiples variables numéricas y categóricas sin necesidad de supuestos estrictos sobre la distribución de los datos, así como su resistencia al sobreajuste mediante el promedio de árboles.

Se adoptó un enfoque riguroso separando los datos en 80% para entrenamiento y 20% para validación, estratificando por la variable objetivo para preservar las proporciones reales en ambos conjuntos.

## 3. Análisis Experimental
El flujo de experimentación se documentó secuencialmente en un conjunto de Jupyter Notebooks:
1. **Análisis Exploratorio (EDA)**: Se analizaron distribuciones y correlaciones para identificar variables con mayor poder discriminante (por ejemplo, `dias_entrega`, `tiempo_estimado_hrs`).
2. **Preprocesamiento**: 
   - Se descartaron columnas identificadoras sin valor predictivo (`id_envio`, `id_vehiculo`, fechas absolutas).
   - Variables categóricas fueron tratadas con One-Hot Encoding.
   - Variables numéricas fueron estandarizadas con `StandardScaler` de Scikit-Learn para acelerar la convergencia y mejorar el rendimiento del modelo logístico.
3. **Entrenamiento y Configuración**: Se utilizaron semillas (`random_state=42`) en cada fase (división de datos y algoritmos) para asegurar el cumplimiento del principio de **reproducibilidad total**. 

## 4. Resultados y Comparación de Modelos
En la etapa de evaluación, ambos modelos fueron medidos según Accuracy, Precision, Recall y F1-Score:
- **Regresión Logística**: Logró un desempeño aceptable como línea base, pero con un ligero déficit en la capacidad para captar patrones no lineales complejos (menor F1-Score general).
- **Random Forest**: Superó al modelo lineal en casi todas las métricas, exhibiendo un alto nivel de Precisión y Recall. La Curva ROC demostró un Área Bajo la Curva (AUC) sustancialmente mayor para este modelo, confirmando su superioridad para distinguir entre envíos puntuales y retrasados.

*Las matrices de confusión y gráficos comparativos (como la Curva ROC) se encuentran documentados y serializados en el directorio `results/plots`.*

## 5. Optimización de Hiperparámetros
Para maximizar el rendimiento del mejor modelo (Random Forest), se implementó una búsqueda paramétrica exhaustiva empleando `GridSearchCV` con validación cruzada de 3 pliegues (3-Fold CV). 
- **Espacio de búsqueda**: Se varió el número de árboles (`n_estimators`: 50, 100, 200), profundidad máxima (`max_depth`: None, 10, 20) y tamaño mínimo de división (`min_samples_split`: 2, 5).
- **Métrica objetivo**: F1-Score, seleccionada explícitamente para garantizar un balance adecuado entre Precisión y Recall, vital si existe algún desbalance entre clases.
- **Impacto**: El modelo optimizado demostró una mejora observable, reduciendo los falsos positivos y refinando los límites de decisión. El modelo final fue serializado con `joblib`.

## 6. Conclusiones y Recomendaciones
### Conclusiones
- Se comprobó la viabilidad de utilizar datos operacionales e incidencias previas para prever retrasos con alta precisión.
- La organización modular estructurada en base a las convenciones de *Data Engineering* (Kedro) y *Data Science* (Scikit-Learn modularizado) asegura la mantenibilidad del proyecto.

### Recomendaciones Futuras
1. **Ingeniería de Características (Feature Engineering)**: Sería valioso extraer estacionalidades más granulares (día de la semana, hora del día) de las fechas originales antes de descartarlas.
2. **Despliegue y MLOps**: Se recomienda empaquetar el modelo ganador (`Random_Forest_Optimizado.joblib`) junto con el `StandardScaler` en una API REST (ej. FastAPI o Flask) para consumirlo en la aplicación logística principal de la compañía.
3. **Validación temporal**: Evaluar el desempeño implementando validación *Out-Of-Time*, entrenando con datos de meses anteriores y probando sobre el mes más reciente, replicando un escenario real en producción.
