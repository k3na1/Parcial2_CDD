"""
This is a boilerplate pipeline 'data_transformation'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import integrar_datos, preprocesar_modelo

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=integrar_datos,
            inputs=[
                "envios_inter", 
                "incidencias_inter", 
                "rutas_inter", 
                "vehiculos_inter"
            ],
            outputs="master_table_raw", # Resultado intermedio en memoria
            name="join_datasets_node",
        ),
        node(
            func=preprocesar_modelo,
            inputs="master_table_raw",
            outputs="master_table", # El que definimos en catalog.yml
            name="preprocessing_node",
        ),
    ])