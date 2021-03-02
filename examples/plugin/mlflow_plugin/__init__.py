import os
import mlflow
import yaml


def _set_mlfow_settings(mlflow_config: dict):
    for k, v in mlflow_config.items():
        if k == 'MLFLOW_TRACKING_URI':
            mlflow.set_tracking_uri(v)
        else:
            os.environ[k] = v


def _set_s3_settings(s3_config: dict):
    for k, v in s3_config.items():
        os.environ[k] = v


def _set_local_settings(local_config: dict):
    for k, v in local_config.items():
        if k == 'NO_PROXY':
            os.environ.setdefault('NO_PROXY', '')
            os.environ["NO_PROXY"] += ',' + v
        else:
            os.environ[k] = v


def _determine_config_type(config):
    if 'MLFLOW' in config:
        _set_mlfow_settings(config['MLFLOW'])
    if 'S3' in config:
        _set_s3_settings(config['S3'])
    if 'LOCAL_ENV' in config:
        _set_local_settings(config['LOCAL_ENV'])


def init_mlflow(config_path_or_dict=None, override: dict = None):

    if isinstance(config_path_or_dict, str):
        with open(config_path_or_dict, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    elif isinstance(config_path_or_dict, dict):
        config = config_path_or_dict
    elif not config_path_or_dict:
        raise TypeError()

    if config:
        _determine_config_type(config)
    if override:
        if isinstance(override, dict):
            _determine_config_type(override)
        else:
            raise TypeError()
