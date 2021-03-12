import mlflow
import mlflow.pyfunc
from mlflow_plugin import init_mlflow
from sklearn import datasets


init_mlflow(config_path_or_dict='/srv/users/v126168/mlflow_docker/examples/elasticnet_diabetes/mlflow-config.yaml')

model_name = 'diabetes'
stage = 'Production'
model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_name}/{stage}')

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data

result = model.predict(X)
print(result)
