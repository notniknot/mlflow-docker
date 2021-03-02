from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_categorical_transformer():
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )
    return categorical_transformer
