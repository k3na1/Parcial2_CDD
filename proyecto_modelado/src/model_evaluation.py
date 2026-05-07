"""
Módulo de evaluación de modelos de Machine Learning.
Provee funciones para evaluar el desempeño de los modelos,
generando métricas y visualizaciones comparativas.
"""

import os
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.base import BaseEstimator

def evaluate_model(
    model: BaseEstimator, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evalúa un modelo entrenado utilizando el conjunto de prueba.
    
    Args:
        model (BaseEstimator): Modelo entrenado a evaluar.
        X_test (pd.DataFrame): Datos predictores de prueba.
        y_test (pd.Series): Valores reales de prueba.
        model_name (str): Nombre del modelo para reportes.
        
    Returns:
        Dict[str, float]: Diccionario con las métricas calculadas.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"--- Evaluación: {model_name} ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return metrics

def plot_confusion_matrix(
    model: BaseEstimator, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str, 
    save_dir: str = "results/plots"
) -> None:
    """
    Genera y guarda la matriz de confusión del modelo.
    
    Args:
        model (BaseEstimator): Modelo entrenado.
        X_test (pd.DataFrame): Datos predictores de prueba.
        y_test (pd.Series): Valores reales de prueba.
        model_name (str): Nombre del modelo.
        save_dir (str): Directorio donde guardar el gráfico.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(file_path)
    plt.close()
    
def plot_roc_curve(
    models: Dict[str, BaseEstimator], 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    save_dir: str = "results/plots"
) -> None:
    """
    Genera y guarda la curva ROC comparando múltiples modelos.
    
    Args:
        models (Dict[str, BaseEstimator]): Diccionario con los modelos entrenados.
        X_test (pd.DataFrame): Datos predictores de prueba.
        y_test (pd.Series): Valores reales de prueba.
        save_dir (str): Directorio donde guardar el gráfico.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        # Verificamos si el modelo soporta predict_proba
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        else:
            print(f"El modelo {name} no soporta predict_proba, se omite de la curva ROC.")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC Comparativa')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    file_path = os.path.join(save_dir, 'roc_curve_comparison.png')
    plt.savefig(file_path)
    plt.close()
