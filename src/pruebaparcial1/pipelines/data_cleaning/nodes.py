"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""

import pandas as pd
import numpy as np

def limpiar_datos_maestros(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Función integral de limpieza para AD 1.2"""
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Corrección de tipos mixtos y IDs
    for col in params.get("id_cols", []):
        if col in df.columns:
            # Convertimos a string para evitar decimales en IDs (ej. 1.0 -> "1")
            df[col] = df[col].fillna(0).astype(float).astype(int).astype(str).replace("0", np.nan)

    # 3. Forzar columnas numéricas que vienen como string
    for col in params.get("num_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Estandarización de fechas
    for col in params.get("date_cols", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # 5. Normalización de strings
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.lower().str.strip().replace("nan", np.nan)

    # 6. Tratamiento de nulos
    # Numéricos: Mediana | Categóricos: 'desconocido'
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('desconocido')

    # 7. Outliers con Rango Intercuartílico (IQR)
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Aplicamos clipping para no perder datos, solo suavizarlos
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df