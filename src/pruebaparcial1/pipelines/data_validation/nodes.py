"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 1.3.1
"""

import pandas as pd
import numpy as np

def validar_integridad_master(master: pd.DataFrame, envios_old: pd.DataFrame) -> dict:
    """Verificación de integridad post-transformación (AD 1.4)"""
    
    # 1. Comparación Antes/Después (Filas)
    filas_originales = len(envios_old)
    filas_finales = len(master)
    diferencia_filas = filas_originales - filas_finales
    
    # 2. Verificación de Integridad (Unicidad)
    duplicados_id = int(master["id_envio"].duplicated().sum())
    
    # 3. Validación de Esquema (Tipos esperados)
    # Comprobamos que las columnas clave sean numéricas después del escalamiento
    cols_criticas = ["distancia_km", "peso_kg", "eficiencia_peso"]
    esquema_ok = all(pd.api.types.is_numeric_dtype(master[c]) for c in cols_criticas)

    # 4. Chequeo de Nulos Finales
    nulos_totales = int(master.isnull().sum().sum())

    reporte = {
        "status": "SUCCESS" if diferencia_filas == 0 and duplicados_id == 0 else "WARNING",
        "comparacion_antes_despues": {
            "filas_envios_original": filas_originales,
            "filas_master_final": filas_finales,
            "perdida_datos": diferencia_filas
        },
        "integridad": {
            "ids_duplicados_en_master": duplicados_id,
            "esquema_numerico_valido": esquema_ok
        },
        "calidad": {
            "nulos_encontrados": nulos_totales,
            "columnas_finales": master.shape[1]
        }
    }

    print("\n--- REPORTE DE VALIDACIÓN ---")
    print(f"¿Pérdida de filas?: {'No' if diferencia_filas == 0 else 'SÍ: ' + str(diferencia_filas)}")
    print(f"¿IDs duplicados?: {'No' if duplicados_id == 0 else 'SÍ'}")
    print(f"¿Nulos en master?: {nulos_totales}")
    
    return reporte