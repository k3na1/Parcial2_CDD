"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import explorar_datos

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=explorar_datos,
            inputs=["envios_raw", "params:name_envios"],
            outputs="reporte_envios",
            name="ingestion_envios",
        ),
        node(
            func=explorar_datos,
            inputs=["incidencias_raw", "params:name_incidencias"],
            outputs="reporte_incidencias",
            name="ingestion_incidencias",
        ),
        node(
            func=explorar_datos,
            inputs=["rutas_raw", "params:name_rutas"],
            outputs="reporte_rutas",
            name="ingestion_rutas",
        ),
        node(
            func=explorar_datos,
            inputs=["vehiculos_raw", "params:name_vehiculos"],
            outputs="reporte_vehiculos",
            name="ingestion_vehiculos",
        ),
    ])
