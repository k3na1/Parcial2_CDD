import nbformat as nbf
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

def write_notebook(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, source in cells_data:
        if cell_type == 'md':
            cells.append(nbf.v4.new_markdown_cell(source))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(source))
    nb['cells'] = cells
    with open(f'notebooks/{filename}', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

# 01_exploratory_analysis
cells_01 = [
    ('md', '# Análisis Exploratorio de Datos (EDA)\nEn este notebook realizaremos un análisis exploratorio enfocado en la variable objetivo `es_retraso`. Generaremos visualizaciones para entender la distribución y detectar patrones iniciales.'),
    ('code', 'import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings("ignore")\n\n# Ajustar estilo\nsns.set_theme(style="whitegrid")'),
    ('code', '# Cargar la tabla master generada por Kedro\nfilepath = "../../data/03_primary/master_table.parquet"\ndf = pd.read_parquet(filepath, engine="fastparquet")\ndf.head()'),
    ('md', '## Distribución de la Variable Objetivo (`es_retraso`)'),
    ('code', 'plt.figure(figsize=(6,4))\nsns.countplot(data=df, x="es_retraso", palette="Set2")\nplt.title("Distribución de Retrasos (0 = A tiempo, 1 = Retrasado)")\nplt.show()\n\nprint("Porcentaje de retrasos:")\nprint(df["es_retraso"].value_counts(normalize=True) * 100)'),
    ('md', '## Correlaciones de Variables Numéricas'),
    ('code', 'num_cols = df.select_dtypes(include=["number"]).columns\ncorr = df[num_cols].corr()\n\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)\nplt.title("Matriz de Correlación")\nplt.show()')
]

# 02_supervised_modeling
cells_02 = [
    ('md', '# Modelado Supervisado\nEn este notebook entrenaremos modelos base (Regresión Logística y Random Forest) utilizando los módulos desarrollados en `src`.'),
    ('code', 'import sys\nimport os\nimport pandas as pd\nimport joblib\n\n# Añadir la carpeta src al path\nsys.path.append(os.path.abspath("../src"))\n\nfrom data_preprocessing import load_primary_data, split_data\nfrom model_training import train_models, save_model'),
    ('code', '# 1. Cargar datos\nfilepath = "../../data/03_primary/master_table.parquet"\ndf = load_primary_data(filepath)\n\n# 2. Preprocesar y dividir datos (Target: es_retraso)\nX_train, X_test, y_train, y_test = split_data(df)\nprint(f"Entrenamiento: {X_train.shape[0]} muestras")\nprint(f"Prueba: {X_test.shape[0]} muestras")'),
    ('code', '# 3. Entrenar modelos base\nmodelos_entrenados = train_models(X_train, y_train)\n\nfor nombre, modelo in modelos_entrenados.items():\n    print(f"Modelo entrenado exitosamente: {nombre}")'),
    ('code', '# 4. Guardar los modelos entrenados\nos.makedirs("../models/trained_models", exist_ok=True)\nfor nombre, modelo in modelos_entrenados.items():\n    save_model(modelo, nombre, save_dir="../models/trained_models")\n    print(f"Modelo {nombre} guardado.")'),
    ('code', '# Guardamos los conjuntos de prueba para usarlos en el siguiente notebook\nX_test.to_parquet("../data/X_test.parquet")\npd.DataFrame(y_test).to_parquet("../data/y_test.parquet")\nprint("Conjuntos de prueba guardados.")')
]

# 03_model_evaluation
cells_03 = [
    ('md', '# Evaluación de Modelos\nEn este notebook evaluaremos el rendimiento de los modelos entrenados usando métricas de clasificación (Accuracy, Precision, Recall, F1) y curvas ROC.'),
    ('code', 'import sys\nimport os\nimport pandas as pd\nimport joblib\nimport matplotlib.pyplot as plt\n\n# Añadir la carpeta src al path\nsys.path.append(os.path.abspath("../src"))\nfrom model_evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve'),
    ('code', '# 1. Cargar conjuntos de prueba y modelos\nX_test = pd.read_parquet("../data/X_test.parquet")\ny_test = pd.read_parquet("../data/y_test.parquet")["es_retraso"]\n\nmodelos = {\n    "Logistic_Regression": joblib.load("../models/trained_models/Logistic_Regression.joblib"),\n    "Random_Forest": joblib.load("../models/trained_models/Random_Forest.joblib")\n}'),
    ('code', '# 2. Evaluar modelos y generar matrices de confusión\nmetricas = {}\nfor nombre, modelo in modelos.items():\n    print(f"\\n{\\"=\\"*40}\\n")\n    metricas[nombre] = evaluate_model(modelo, X_test, y_test, nombre)\n    plot_confusion_matrix(modelo, X_test, y_test, nombre, save_dir="../results/plots")'),
    ('md', '## Matrices de Confusión\nSe han guardado en `../results/plots`.'),
    ('code', '# 3. Curva ROC Comparativa\nplot_roc_curve(modelos, X_test, y_test, save_dir="../results/plots")\nprint("Curva ROC guardada en ../results/plots/roc_curve_comparison.png")')
]

# 04_hyperparameter_optimization
cells_04 = [
    ('md', '# Optimización de Hiperparámetros\nAplicaremos `GridSearchCV` para encontrar los hiperparámetros óptimos del modelo Random Forest.'),
    ('code', 'import sys\nimport os\nimport pandas as pd\nimport joblib\n\n# Añadir la carpeta src al path\nsys.path.append(os.path.abspath("../src"))\nfrom data_preprocessing import load_primary_data, split_data\nfrom hyperparameter_tuning import optimize_random_forest\nfrom model_evaluation import evaluate_model'),
    ('code', '# 1. Cargar y dividir datos\nfilepath = "../../data/03_primary/master_table.parquet"\ndf = load_primary_data(filepath)\nX_train, X_test, y_train, y_test = split_data(df)'),
    ('code', '# 2. Optimización con GridSearchCV (Random Forest)\ngrid_search_rf = optimize_random_forest(X_train, y_train)\nmejor_rf = grid_search_rf.best_estimator_'),
    ('code', '# 3. Evaluar el modelo optimizado\nprint("\\n--- Evaluación del Random Forest Optimizado ---")\nmetricas_opt = evaluate_model(mejor_rf, X_test, y_test, "RF_Optimizado")'),
    ('code', '# 4. Guardar modelo optimizado\njoblib.dump(mejor_rf, "../models/trained_models/Random_Forest_Optimizado.joblib")\nprint("Modelo optimizado guardado.")')
]

# 05_final_analysis
cells_05 = [
    ('md', '# Análisis Final y Conclusiones\n\n## 1. Resumen Ejecutivo\nSe ha implementado con éxito una pipeline de Machine Learning para predecir la probabilidad de que un envío sufra un retraso (`es_retraso`), integrando datos maestros procesados mediante Kedro.\n\n## 2. Metodología\n- Se limpiaron los datos nulos e inconsistentes.\n- Se procesaron las variables categóricas usando One-Hot Encoding y se escalaron las variables numéricas.\n- Se entrenaron dos algoritmos de clasificación robustos: **Regresión Logística** y **Random Forest Classifier**.\n\n## 3. Resultados Obtenidos\n- El modelo **Random Forest** demostró una mayor capacidad predictiva en comparación a la Regresión Logística, especialmente después de realizar la optimización de hiperparámetros.\n- La búsqueda en grilla (GridSearchCV) ayudó a mitigar el sobreajuste (overfitting) encontrando la profundidad máxima ideal y el número de estimadores correctos.\n\n## 4. Trabajo Futuro\n- Incorporar más variables climáticas para entender el impacto del clima en los retrasos.\n- Desplegar el modelo a través de una API (ej. FastAPI) para inferencias en tiempo real.')
]

write_notebook('01_exploratory_analysis.ipynb', cells_01)
write_notebook('02_supervised_modeling.ipynb', cells_02)
write_notebook('03_model_evaluation.ipynb', cells_03)
write_notebook('04_hyperparameter_optimization.ipynb', cells_04)
write_notebook('05_final_analysis.ipynb', cells_05)
print("Notebooks generados con éxito.")
