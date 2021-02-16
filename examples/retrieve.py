import mlflow
import os

import mlflow.pyfunc
import mlflow
import pandas as pd

REMOTE_IP = '127.0.0.1'
os.environ['no_proxy'] = REMOTE_IP

os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

os.environ["AWS_ACCESS_KEY_ID"] = "mein_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mein_secret_key"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{REMOTE_IP}:9000/"

mlflow.set_tracking_uri(f"http://{REMOTE_IP}:5000")

mlflow.set_experiment("exp")


def get_data():
    URL = 'examples/processed.cleveland.data'
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


model_name = "test_model"
stage = 'Staging'


model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

print(model.predict(pd.read_feather('examples/data/val.feather')))
