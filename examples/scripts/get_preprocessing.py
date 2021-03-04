import pandas as pd
from mlflow_plugin import init_mlflow
from mlflow_plugin.func import get_preprocessing

init_mlflow(config_path_or_dict='/home/niklas/Documents/mlflow_docker/examples/mlflow-config.yaml')

model_name = 'test_model'
stage = 'Staging'
paths = get_preprocessing(
    model_uri=f'models:/{model_name}/{stage}',
    preprocessing_location='preprocessing',
    artifacts=[''],
)

df = pd.read_feather(paths['artifacts'][''] + '/val.feather')

from preprocessing.pipeline import get_pipeline

print(get_pipeline())
