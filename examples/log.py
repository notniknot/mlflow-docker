import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers.pipeline import get_pipeline


def get_data():
    URL = './examples/processed.cleveland.data'
    df = pd.read_csv(
        URL,
        header=None,
        names=[
            'age',
            'sex',
            'cp',
            'trestbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal',
            'num',
        ],
    )
    df['target'] = np.where(df['num'] > 0, 1, 0)

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'Train Examples')
    print(len(val), 'Validation Examples')
    print(len(test), 'Test Examples')

    return train, val, test


def main():

    train, val, test = get_data()

    pipline = get_pipeline()

    # fit the pipeline
    pipline.fit(train, train['target'].values)

    # create predictions for validation data
    y_pred = pipline.predict(val)

    class ModelOut(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model

        def load_context(self, context):
            print(context.artifacts)
            print('load_context')

        def predict(self, context, model_input):
            model_input.columns = map(str.lower, model_input.columns)
            print('predict')
            return self.model.predict_proba(model_input)[:, 1]

    mlflow_conda = {
        'channels': ['defaults'],
        'name': 'conda',
        'dependencies': [
            'python=3.6',
            'pip',
            {'pip': ['mlflow', 'scikit-learn', 'cloudpickle', 'pandas', 'numpy']},
        ],
    }

    REMOTE_IP = '127.0.0.1'
    os.environ['no_proxy'] = REMOTE_IP

    os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

    os.environ["AWS_ACCESS_KEY_ID"] = "mein_access_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "mein_secret_key"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{REMOTE_IP}:9000/"

    mlflow.set_tracking_uri(f"http://{REMOTE_IP}:5000")
    mlflow.set_experiment("exp")

    with mlflow.start_run(run_name='test7') as run:
        # log metrics
        mlflow.log_metric("accuracy", accuracy_score(val['target'].values, y_pred))
        mlflow.log_metric("precison", precision_score(val['target'].values, y_pred))
        mlflow.log_metric("recall", recall_score(val['target'].values, y_pred))

        # log model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ModelOut(model=pipline),
            code_path=['examples/transformers'],
            conda_env=mlflow_conda,
            artifacts={'val.feather': 'examples/data/val.feather'},
        )
        # signature = infer_signature(val, y_pred)

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri, "test_model")

    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))


if __name__ == '__main__':
    main()
