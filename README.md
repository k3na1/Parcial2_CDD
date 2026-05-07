# Proyecto de Modelado - Parcial 2 (Ciencia de Datos)

Este repositorio contiene la implementación completa de un pipeline de Machine Learning integrado con **Kedro** para la predicción de retrasos en envíos (`es_retraso`). 

El proyecto consta de dos partes principales:
1. **Data Engineering (Kedro)**: Nodos para la limpieza e integración de múltiples fuentes de datos (`envios`, `incidencias`, `rutas`, `vehiculos`) en una única `master_table.parquet`.
2. **Machine Learning (proyecto_modelado)**: Scripts y Notebooks que realizan el entrenamiento, evaluación y optimización de modelos (Random Forest y Regresión Logística) sobre la tabla maestra.

---

## 🚀 Cómo inicializar el proyecto

Para asegurar la reproducibilidad y evitar problemas de compatibilidad (como el `KedroPythonVersionWarning`), el proyecto debe ejecutarse estrictamente con **Python 3.10.6**.

### 1. Clonar el repositorio
```bash
git clone https://github.com/k3na1/Parcial2_CDD.git
cd Parcial2_CDD
```

### 2. Crear y activar el entorno virtual
Es altamente recomendable utilizar `uv` para la gestión del entorno:
```bash
uv venv --python 3.10.6
# En Windows:
.venv\Scripts\activate
# En Mac/Linux:
source .venv/bin/activate
```

### 3. Instalar las dependencias
Todas las dependencias necesarias para Kedro y Scikit-Learn están listadas.
```bash
uv pip install -e .
uv pip install pandas numpy scikit-learn matplotlib seaborn joblib pyarrow fastparquet
```
*(También puedes usar `pip install` tradicional si no utilizas `uv`).*

---

## ⚙️ Generación de Datos (Kedro Run)

Antes de entrenar los modelos, necesitas generar la tabla maestra procesada. Ejecuta el pipeline de Kedro:

```bash
kedro run
```
Esto procesará los CSV en crudo y generará el archivo `data/03_primary/master_table.parquet`.

---

## 🧠 ¿Cómo ver lo que se hizo en la Parcial 2 (Machine Learning)?

Todo el código de modelado se estructuró dentro de la carpeta `proyecto_modelado/`.

### Opción A: Explorar los Jupyter Notebooks
Se ha construido un flujo de análisis en 5 cuadernos que debes ejecutar de forma secuencial. Puedes abrirlos usando VS Code o lanzando Jupyter Lab:
```bash
jupyter lab
```
Dirígete a `proyecto_modelado/notebooks/` y revisa:
1. **`01_exploratory_analysis.ipynb`**: Análisis descriptivo y distribuciones de `es_retraso`.
2. **`02_supervised_modeling.ipynb`**: Separación de datos (entrenamiento/prueba) y ajuste de modelos base.
3. **`03_model_evaluation.ipynb`**: Métricas de rendimiento, Curva ROC y Matrices de Confusión.
4. **`04_hyperparameter_optimization.ipynb`**: Tuning de hiperparámetros con `GridSearchCV` para Random Forest.
5. **`05_final_analysis.ipynb`**: Análisis de importancia de variables (*Feature Importance*) y conclusiones.

### Opción B: Explorar los Scripts (Archivos `.py`)
El código fuente utilizado por los notebooks se encuentra modularizado en `proyecto_modelado/src/`:
- `data_preprocessing.py`: Carga y escalado de datos con `StandardScaler`.
- `model_training.py`: Instanciación y entrenamiento de modelos.
- `model_evaluation.py`: Generación de gráficos y métricas.
- `hyperparameter_tuning.py`: Lógica de optimización.

### Opción C: Informe Técnico Final
Para leer el resumen metodológico y las conclusiones de todo este proceso experimental, dirígete al archivo **[`proyecto_modelado/informe_tecnico.md`](proyecto_modelado/informe_tecnico.md)**.
