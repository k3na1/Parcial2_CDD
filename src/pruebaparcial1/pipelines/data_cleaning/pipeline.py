"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import limpiar_datos_maestros

def create_pipeline(**kwargs) -> Pipeline:
    datasets = ["envios", "incidencias", "rutas", "vehiculos"]
    nodes = []
    
    for ds in datasets:
        nodes.append(
            node(
                func=limpiar_datos_maestros,
                inputs=[f"{ds}_raw", f"params:cleaning_params.{ds}"],
                outputs=f"{ds}_inter",
                name=f"cleaning_node_{ds}",
            )
        )
    
    return pipeline(nodes)