import os
import gzip
import pickle

from attribmod.utils import datafilefinder


def get_encoder() -> dict:
    if 'MLFLOW_ARTIFACT_ENCODER' in os.environ:
        filepath = os.getenv('MLFLOW_ARTIFACT_ENCODER')
    else:
        filepath = datafilefinder.get_path('models', 'encoder.pickle.gz')
    with gzip.open(filepath, mode='rb') as file:
        _, encoder = pickle.load(file)
    return encoder


def get_missing() -> str:
    if 'MLFLOW_ARTIFACT_ENCODER' in os.environ:
        filepath = os.getenv('MLFLOW_ARTIFACT_ENCODER')
    else:
        filepath = datafilefinder.get_path('models', 'encoder.pickle.gz')
    with gzip.open(filepath, mode='rb') as file:
        missing, _ = pickle.load(file)
    return missing


def store_encoder(content: list):
    if 'MLFLOW_ARTIFACT_ENCODER' in os.environ:
        filepath = os.getenv('MLFLOW_ARTIFACT_ENCODER')
    else:
        filepath = datafilefinder.get_path('models', 'encoder.pickle.gz')
    with gzip.open(filepath, mode='wb') as file:
        pickle.dump(content, file)
