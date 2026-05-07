"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""

import pandas as pd
import logging

def explorar_datos(df: pd.DataFrame, nombre: str) -> dict:
    logger = logging.getLogger(__name__)
    logger.info(f"Explorando dataset: {nombre}")

    reporte = {
        "dataset": nombre,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "tipos": df.dtypes.astype(str).to_dict(),
        "nulos": df.isnull().sum().to_dict(),
        "duplicados": int(df.duplicated().sum()),
        "resumen": df.describe(include='all').to_dict()
    }
    
    # Imprimimos en consola para cumplir con la "Exploración inicial"
    print(f"\n--- REPORTE: {nombre} ---")
    print(df.info())
    print(df.head())
    
    return reporte