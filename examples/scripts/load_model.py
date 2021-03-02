import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow_plugin import init_mlflow

init_mlflow(config_path_or_dict='/home/niklas/Documents/mlflow_docker/examples/mlflow-config.yaml')

model_name = 'test_model'
stage = 'Staging'
model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{stage}')

print(model.predict(pd.read_feather('examples/data/val.feather')))
