from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing.custom_transformer import CustomTransformer
from preprocessing.numeric_transformer import get_numeric_transformer
from preprocessing.categorical_transformer import get_categorical_transformer


def get_pipeline():
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    numeric_transformer = get_numeric_transformer()

    categorical_features = ['cp', 'restecg', 'ca', 'thal', 'slope']
    categorical_transformer = get_categorical_transformer()

    binary_features = ['sex', 'fbs', 'exang']
    binary_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    new_features_input = ['thalach', 'trestbps']
    new_transformer = Pipeline(steps=[('custom_transformer', CustomTransformer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('binary', binary_transformer, binary_features),
            ('new', new_transformer, new_features_input),
        ]
    )

    clf = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())], verbose=True
    )

    return clf
