import os
import mlflow
import pandas as pd

from attribmod.utils import configuration, my_logging, my_metrics

LOGGER = my_logging.logger(os.path.basename(__file__))

_myflow = None


def mltracker() -> mlflow:
    global _myflow
    if _myflow:
        return _myflow
    uri = configuration.get_value('mltracker', 'uri')
    project_name = configuration.get_value('mltracker', 'project_name')
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(project_name)
    _myflow = mlflow
    return mlflow


def start_run():
    return mltracker().start_run()


def log_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, logging=True):
    metrics = my_metrics.compute_metrics(y_true, y_pred)

    mltracker().log_metrics(metrics)

    if logging:
        for metric, value in metrics.items():
            LOGGER.info(f"{metric}: {value}")
