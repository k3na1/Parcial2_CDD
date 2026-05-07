"""
Módulo de entrenamiento de modelos de Machine Learning.
Contiene funciones para instanciar, entrenar y guardar modelos predictivos
utilizando buenas prácticas y permitiendo reproducibilidad.
"""

import os
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

def train_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    random_state: int = 42
) -> Dict[str, BaseEstimator]:
    """
    Entrena modelos base predefinidos utilizando los datos de entrenamiento.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        random_state (int): Semilla para asegurar la reproducibilidad.
        
    Returns:
        Dict[str, BaseEstimator]: Diccionario con los nombres de los modelos 
                                  y las instancias de los modelos entrenados.
                                  
    Raises:
        ValueError: Si los datos de entrada están vacíos o las formas no coinciden.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("X_train o y_train no pueden estar vacíos.")
        
    if len(X_train) != len(y_train):
        raise ValueError("X_train y y_train deben tener la misma cantidad de filas.")

    models = {
        "Logistic_Regression": LogisticRegression(
            random_state=random_state, 
            max_iter=1000, 
            class_weight='balanced'
        ),
        "Random_Forest": RandomForestClassifier(
            random_state=random_state, 
            n_estimators=100, 
            class_weight='balanced'
        )
    }

    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            raise RuntimeError(f"Error entrenando el modelo {name}: {str(e)}")

    return trained_models

def save_model(model: BaseEstimator, model_name: str, save_dir: str = "models/trained_models") -> str:
    """
    Serializa y guarda un modelo entrenado en disco.
    
    Args:
        model (BaseEstimator): Modelo de sklearn entrenado.
        model_name (str): Nombre del modelo para nombrar el archivo.
        save_dir (str): Directorio donde guardar el modelo.
        
    Returns:
        str: Ruta completa donde se guardó el modelo.
        
    Raises:
        NotFittedError: Si se intenta guardar un modelo que no ha sido entrenado.
        IOError: Si ocurre un problema al guardar el archivo.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}.joblib")
    
    try:
        # joblib es eficiente para serializar objetos que contienen grandes arrays numpy (como los modelos de scikit-learn)
        joblib.dump(model, file_path)
    except Exception as e:
        raise IOError(f"No se pudo guardar el modelo {model_name} en {file_path}: {str(e)}")
        
    return file_path
