"""
Módulo de preprocesamiento de datos para modelado de Machine Learning.
Contiene las funciones necesarias para cargar la tabla primaria, separar la 
variable objetivo (es_retraso) y dividir los datos para entrenamiento y prueba.
"""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_primary_data(filepath: str) -> pd.DataFrame:
    """
    Carga la tabla master generada por Kedro desde un archivo Parquet.
    
    Args:
        filepath (str): Ruta al archivo parquet de la tabla master.
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta proporcionada.
        ValueError: Si el archivo cargado no tiene datos o no es válido.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"El archivo {filepath} no existe. "
            "Asegúrate de ejecutar 'kedro run' primero para generar los datos."
        )
    
    try:
        # Se fuerza el motor fastparquet para evitar errores de incompatibilidad de pyarrow (py_extension_type)
        df = pd.read_parquet(filepath, engine='fastparquet')
    except Exception as e:
        raise ValueError(f"Error al leer el archivo parquet {filepath}: {str(e)}")
        
    if df.empty:
        raise ValueError("El archivo cargado está vacío.")
        
    return df

def split_data(
    df: pd.DataFrame, 
    target_col: str = "es_retraso", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa el DataFrame en variables predictoras (X) y variable objetivo (y),
    y luego realiza la división en conjuntos de entrenamiento y prueba.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos procesados.
        target_col (str): Nombre de la columna objetivo.
        test_size (float): Proporción del dataset para pruebas.
        random_state (int): Semilla para reproducibilidad.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
            
    Raises:
        KeyError: Si la columna objetivo no se encuentra en el DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")
        
    X = df.drop(columns=[target_col])
    # Se eliminan también columnas identificadoras que no sirven para el modelo
    cols_to_drop = ["id_envio", "id_ruta", "id_vehiculo", "fecha_envio", "fecha_entrega"]
    cols_present = [c for c in cols_to_drop if c in X.columns]
    X = X.drop(columns=cols_present)
    
    # Codificar variables categóricas (strings) a numéricas (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Rellenar valores nulos (si los hay) para que los modelos no fallen
    X = X.fillna(0)
    
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Buena práctica: Escalar los datos para modelos lineales (evita el ConvergenceWarning de LogisticRegression)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
