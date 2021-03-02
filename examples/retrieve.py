import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow_plugin import init_mlflow
from mlflow_plugin.func import get_preprocessing

init_mlflow(
    config_path_or_dict=r'C:\Users\Niklas\Documents\Projects\mlflow_env\examples\mlflow-config.yaml'
)

# import os
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'mlflow'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'mlflow'
# os.environ['AWS_ACCESS_KEY_ID'] = 'mein_access_key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'mein_secret_key'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:19000/'
# os.environ['MLFLOW_DST_PATH'] = r'C:\Users\Niklas\Documents\Projects\mlflow_env\examples\artifacts'
# mlflow.set_tracking_uri('http://127.0.0.1:15000')


model_name = 'test_model'
stage = 'Staging'

paths = get_preprocessing(
    model_uri=f'models:/{model_name}/{stage}',
    preprocessing_location='preprocessing',
    dst_path=r'D:\Temp\mlflow_preprocessing',
    artifacts=[''],
)

# df = pd.read_feather(paths['artifacts'][''] + '/val.feather')

# from preprocessing.pipeline import get_pipeline

# print(get_pipeline())

# import importlib.util
# spec = importlib.util.spec_from_file_location("get_pipeline", output + '/' + "pipeline.py")
# foo = importlib.util.module_from_spec(spec)
# print(spec.loader.exec_module(foo))
# print(foo.get_pipeline())


model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{stage}')

print(model.predict(pd.read_feather('examples/data/val.feather')))
