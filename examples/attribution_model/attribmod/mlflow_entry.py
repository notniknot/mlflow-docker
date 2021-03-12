import os

import mlflow
from attribmod.data.data_generation import orchestrate_data_generation
from attribmod.features.sessions_to_cjs import orchestrate_data_processing
from attribmod.models.model_application import orchestrate_model_application
from attribmod.utils.env import check_env


class ModelOut(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        for artifact, path in context.artifacts.items():
            os.environ[f'MLFLOW_ARTIFACT_{artifact.upper()}'] = path
        check_env()

    def predict(self, context, model_input):
        input_data = model_input['input_data']
        from_date = model_input['from_date']
        to_date = model_input['to_date']
        path_to_bs = orchestrate_data_generation(
            input_data,
            from_date,
            to_date,
        )
        path_to_abt = orchestrate_data_processing(path_to_bs=path_to_bs)
        path_to_result = orchestrate_model_application(path_to_abt=path_to_abt)
        return path_to_result
