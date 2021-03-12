import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow_plugin import init_mlflow

init_mlflow(config_path_or_dict='/srv/users/v126168/mlflow_docker/examples/attribution_model/mlflow-config.yaml')

model_name = 'attribution_model'
stage = 'Staging'
model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{stage}')

jan_bs_raw = pd.read_feather('/srv/users/v126168/mlflow_docker/examples/attribution_model/data_df.feather')
jan_bs_raw = jan_bs_raw.iloc[:1000]

path_to_result = model.predict(
    {'input_data': jan_bs_raw, 'from_date': '2021-01-01', 'to_date': '2021-01-31'}
)

print(pd.read_feather(path_to_result).head())
