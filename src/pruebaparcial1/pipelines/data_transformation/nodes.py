"""
This is a boilerplate pipeline 'data_transformation'
generated using Kedro 1.3.1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def integrar_datos(envios: pd.DataFrame, incidencias: pd.DataFrame, 
                   rutas: pd.DataFrame, vehiculos: pd.DataFrame) -> pd.DataFrame:
    """Realiza joins y transformaciones avanzadas (AD 1.3)"""
    
    # 1. Transformación Avanzada: Groupby + Pivot en Incidencias
    # Agrupamos por envío para tener un resumen de costos y tipos de incidencias
    resumen_incidencias = incidencias.groupby("id_envio").agg(
        total_costo_incidencias=("costo_impacto", "sum"),
        n_incidencias=("id_incidencia", "count")
    ).reset_index()

    # 2. Joins entre las 4 tablas (Capa Master)
    # Empezamos con envios (tabla principal) y unimos el resto
    master = envios.merge(rutas, on="id_ruta", how="left")
    master = master.merge(vehiculos, on="id_vehiculo", how="left")
    master = master.merge(resumen_incidencias, on="id_envio", how="left")

    # Rellenamos nulos en incidencias (los que no tuvieron incidentes tienen costo 0)
    master["total_costo_incidencias"] = master["total_costo_incidencias"].fillna(0)
    master["n_incidencias"] = master["n_incidencias"].fillna(0)

    # 3. Creación de Features Derivadas
    # Eficiencia de carga (peso vs capacidad del vehículo)
    master["eficiencia_peso"] = master["peso_kg"] / master["capacidad_kg"]
    
    # Tiempo real de entrega (en días)
    master["dias_entrega"] = (master["fecha_entrega"] - master["fecha_envio"]).dt.days
    
    # Flag de retraso (si días entrega > tiempo estimado convertido a días)
    master["es_retraso"] = (master["dias_entrega"] > (master["tiempo_estimado_hrs"] / 24)).astype(int)

    return master

def preprocesar_modelo(master: pd.DataFrame) -> pd.DataFrame:
    """Versión robusta con depuración de columnas"""
    
    # --- PASO DE DEPURACIÓN (MIRA TU TERMINAL AL CORRERLO) ---
    print("\n" + "="*30)
    print("DEBUG: Columnas detectadas en master_table:")
    print(master.columns.tolist())
    print("="*30 + "\n")

    # 1. Parche para Parquet (el que vimos antes)
    for col in master.select_dtypes(include=['object']).columns:
        master[col] = master[col].astype(str).replace("nan", np.nan)

    # 2. Codificación SEGURA (Solo codifica si la columna existe)
    cols_a_codificar = ["tipo_carga", "tipo_via", "estado_vehiculo"]
    # Filtramos la lista: solo nos quedamos con las que sí están en el DataFrame
    cols_presentes = [c for c in cols_a_codificar if c in master.columns]
    
    if cols_presentes:
        master = pd.get_dummies(master, columns=cols_presentes, drop_first=True)
    else:
        print("ADVERTENCIA: No se encontró ninguna columna para codificar.")

    # 3. Normalización/Estandarización (También con chequeo)
    num_cols = ["distancia_km", "peso_kg", "total_costo_incidencias", "eficiencia_peso"]
    num_cols_presentes = [c for c in num_cols if c in master.columns]
    
    if num_cols_presentes:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        master[num_cols_presentes] = scaler.fit_transform(master[num_cols_presentes].fillna(0))

    return master