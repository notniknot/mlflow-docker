FROM python:3.8-slim-buster

# Install python packages
RUN pip install --no-cache-dir mlflow==1.13.1 psycopg2-binary==2.8.6 boto3==1.17.5

# Uncomment if using the mlflow-extension
# COPY mlflow_extension-1.0.1-py3-none-any.whl .
# RUN pip install --no-cache-dir mlflow_extension-1.0.1-py3-none-any.whl