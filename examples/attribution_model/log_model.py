import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow_plugin import init_mlflow

from attribmod.mlflow_entry import ModelOut


def main():
    # mlflow_conda = {
    #     'channels': ['defaults'],
    #     'name': 'conda',
    #     'dependencies': [
    #         'python=3.6',
    #         'pip',
    #         {'pip': ['mlflow', 'scikit-learn', 'cloudpickle', 'pandas', 'numpy']},
    #     ],
    # }

    init_mlflow(
        config_path_or_dict='/srv/users/v126168/mlflow_docker/examples/attribution_model/mlflow-config.yaml',
        experiment="attribution_model",
    )

    with mlflow.start_run(run_name='model_run1') as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ModelOut(),
            code_path=['examples/attribution_model/attribmod'],
            # conda_env=mlflow_conda,
            artifacts={
                'encoder': 'examples/attribution_model/models/encoder.pickle.gz',
                'convmod_classifier': 'examples/attribution_model/models/convmod_classifier.pickle.gz',
                'fsc': 'examples/attribution_model/models/fsc.pickle.gz',
            },
            # signature=infer_signature(val, y_pred),
        )

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, "attribution_model")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))


if __name__ == '__main__':
    main()
