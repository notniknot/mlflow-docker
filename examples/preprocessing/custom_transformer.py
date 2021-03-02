import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X['ratio'] = X['thalach'] / X['trestbps']
        X = pd.DataFrame(X.loc[:, 'ratio'])
        return X.values
