from mlflow.tracking import MlflowClient
from mlflow_plugin import init_mlflow

init_mlflow(config_path_or_dict='/home/niklas/Documents/mlflow_docker/examples/mlflow-config.yaml')

client = MlflowClient()
client.transition_model_version_stage(
    name="test_model", version=14, stage="Staging", archive_existing_versions=True
)
