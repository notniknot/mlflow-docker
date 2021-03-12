import os
import pickle

import numpy as np
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeProductPage(TransformerMixin, BaseEstimator):
    def __init__(self, encoder=None, missing=None):
        self.encoder = encoder
        self.missing = missing

    def fit(self, X, y=None):
        return self

    def _provide_fitted_encoder(self, X):
        if self.encoder is None:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(
                np.array([self.missing] + X['produktseite'].unique().tolist()).reshape(-1, 1)
            )
        else:
            self.encoder = pickle.loads(self.encoder)
            # Set unknown values in encoder-categories to missing
            X.loc[
                ~X['produktseite'].isin(self.encoder.categories_[0]), 'produktseite'
            ] = self.missing

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')

        X["produktseite"] = X["produktseite"].str.split(",")
        X["number_products"] = X["produktseite"].apply(lambda x: len(x))
        X["produktseite"] = X["produktseite"].apply(lambda x: x[0])

        self._provide_fitted_encoder(X)
        if self.encoder.inverse_transform([[0]]) != [self.missing]:
            raise ValueError(
                'OrdinalEncoder mapped 0 nicht auf missing, sondern auf %s'
                % self.encoder.inverse_transform([0])[0][0]
            )
        X['produktseite'] = self.encoder.transform(X['produktseite'].values.reshape(-1, 1))
        return X
