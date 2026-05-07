# Proyecto de Modelado: Predicción de Retrasos en Envíos

Este directorio contiene la implementación de los modelos de Machine Learning solicitados para el encargo grupal. 
La estructura del proyecto y los scripts están diseñados siguiendo las mejores prácticas de modularidad, uso eficiente de recursos y reproducibilidad.

## Requisitos y Preparación

Antes de ejecutar los notebooks, es **CRÍTICO** que la tabla maestra de datos esté generada. Este proyecto consume la tabla primaria resultante de las pipelines de Kedro.

Si aún no lo has hecho, asegúrate de correr en la raíz de tu proyecto:
```bash
uv run kedro run
```
Esto generará el archivo `data/03_primary/master_table.parquet`.

## Estructura del Directorio

- `notebooks/`: Contiene los 5 Jupyter Notebooks secuenciales con el análisis, entrenamiento, evaluación y optimización.
- `src/`: Contiene los scripts fuente de Python con código modular, documentado (docstrings) y manejo de excepciones.
  - `data_preprocessing.py`: Carga y división de datos.
  - `model_training.py`: Entrenamiento y guardado de modelos base.
  - `model_evaluation.py`: Generación de métricas, matriz de confusión y curva ROC.
  - `hyperparameter_tuning.py`: Búsqueda de los mejores parámetros (GridSearchCV).
- `models/trained_models/`: Directorio donde se guardan los modelos `.joblib` para la reproducibilidad.
- `results/`: Resultados experimentales organizados en subcarpetas (`metrics`, `plots`, `reports`).
- `informe_tecnico.md`: Plantilla estructurada para el informe técnico de 12-15 páginas.

## Secuencia de Ejecución

Para una reproducibilidad al 100%, sigue este orden en los notebooks:
1. `01_exploratory_analysis.ipynb`: Ejecuta para explorar la tabla.
2. `02_supervised_modeling.ipynb`: Entrena Random Forest y Regresión Logística.
3. `03_model_evaluation.ipynb`: Compara el desempeño inicial y genera los gráficos.
4. `04_hyperparameter_optimization.ipynb`: Optimiza Random Forest con GridSearchCV.
5. `05_final_analysis.ipynb`: Analiza el resultado del mejor modelo.

## Buenas Prácticas Implementadas
- **Reproducibilidad:** Uso de `random_state=42` en la división de datos, Random Forest y Regresión Logística.
- **Manejo Robusto de Excepciones:** Validaciones al cargar el parquet (en caso de que Kedro no haya corrido) y prevención de errores de dimensiones.
- **Modularidad:** Código DRY (Don't Repeat Yourself) reutilizando componentes a través de importaciones limpias.
- **Eficiencia Computacional:** Configuración de `n_jobs=-1` y `cv=3` en la búsqueda de grilla para minimizar el tiempo de cómputo en la optimización.
