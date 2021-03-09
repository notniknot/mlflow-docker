from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_binary_transformer():
    binary_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )
    return binary_transformer
