"""
Módulo de optimización de hiperparámetros.
Contiene funciones para realizar la búsqueda de hiperparámetros óptimos
mediante GridSearchCV, asegurando un uso eficiente de recursos computacionales.
"""

from typing import Dict, Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def optimize_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> GridSearchCV:
    """
    Realiza una búsqueda exhaustiva (GridSearchCV) para optimizar
    los hiperparámetros de un modelo RandomForest.
    
    Args:
        X_train (pd.DataFrame): Datos predictores de entrenamiento.
        y_train (pd.Series): Valores objetivo de entrenamiento.
        random_state (int): Semilla aleatoria para la reproducibilidad.
        
    Returns:
        GridSearchCV: Objeto GridSearchCV ajustado con el mejor modelo.
        
    Raises:
        ValueError: Si los datos de entrada están vacíos.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("X_train o y_train no pueden estar vacíos.")
        
    # Definición del modelo base
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    
    # Grilla de hiperparámetros a explorar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Configuración de GridSearchCV usando validación cruzada y procesadores en paralelo
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3,                 # k-fold cross validation (k=3 para menor costo computacional)
        scoring='f1',         # Optimizamos sobre F1-Score por posible desbalance
        n_jobs=-1,            # Uso de todos los núcleos disponibles
        verbose=1
    )
    
    print("Iniciando optimización de hiperparámetros para RandomForest...")
    grid_search.fit(X_train, y_train)
    print("Optimización completada.")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score (F1): {grid_search.best_score_:.4f}")
    
    return grid_search
