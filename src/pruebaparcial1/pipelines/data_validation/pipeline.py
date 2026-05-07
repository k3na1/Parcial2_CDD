"""
This is a boilerplate pipeline 'data_validation'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import validar_integridad_master

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=validar_integridad_master,
            inputs=["master_table", "envios_inter"], # Compara Primary vs Intermediate
            outputs="reporte_validacion",
            name="nodo_validacion_integridad",
        ),
    ])