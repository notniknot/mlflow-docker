import pandas as pd
from mlflow_plugin import init_mlflow
from mlflow_plugin.func import get_preprocessing

init_mlflow(config_path_or_dict='/srv/users/v126168/mlflow_docker/examples/attribution_model/mlflow-config.yaml')

model_name = 'attribution_model'
stage = 'Staging'
paths = get_preprocessing(
    model_uri=f'models:/{model_name}/{stage}',
    preprocessing_location='attribmod',
    artifacts=['encoder.pickle.gz'],
)

jan_bs_raw = pd.read_feather('/srv/users/v126168/mlflow_docker/examples/attribution_model/data_df.feather')
jan_bs_raw = jan_bs_raw.iloc[:1000]

from attribmod.data.data_generation import orchestrate_data_generation
from attribmod.utils.env import check_env

check_env()
path_to_bs = orchestrate_data_generation(
    jan_bs_raw,
    '2021-01-01',
    '2021-01-31',
)
