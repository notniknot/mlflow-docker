import os
import pickle

import numpy as np
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeFirstPage(TransformerMixin, BaseEstimator):
    def __init__(self, encoder=None, missing=None):
        self.encoder = encoder
        self.missing = missing

    def fit(self, X, y=None):
        return self

    def _provide_fitted_encoder(self, X, i):
        if self.encoder is None:
            self.encoder = [None] * 6
        elif isinstance(self.encoder, bytes):
            self.encoder = pickle.loads(self.encoder)
        if isinstance(self.encoder, list):
            if self.encoder[i] is None:
                self.encoder[i] = OrdinalEncoder()
                self.encoder[i].fit(
                    np.array([self.missing] + X[f'first_page_{i}'].unique().tolist()).reshape(-1, 1)
                )
            else:
                # Set unknown values in encoder-categories to missing
                X.loc[
                    ~X[f'first_page_{i}'].isin(self.encoder[i].categories_[0]), f'first_page_{i}'
                ] = self.missing

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')

        # Split page names, take the first 6 items and fill with missing if needed
        X['first_page_split'] = X['first_page'].astype('str').fillna(self.missing).str.split('_')
        X['first_page_0_5'] = X['first_page_split'].apply(
            lambda l: l[:6] + [self.missing] * (6 - len(l))
        )

        # Spread page splits across 6 columns
        for i in range(6):
            X[f'first_page_{i}'] = (
                X['first_page_0_5']
                .apply(lambda l: l[i] if len(l[i]) > 0 else 'LEER')
                .fillna(self.missing)
            )

            self._provide_fitted_encoder(X, i)
            if self.encoder[i].inverse_transform([[0]]) != self.missing:
                raise ValueError(
                    'OrdinalEncoder mapped 0 nicht auf missing, sondern auf "%s"'
                    % self.encoder[i].inverse_transform([[0]])[0][0]
                )

            X[f'first_page_{i}'] = self.encoder[i].transform(
                X[f'first_page_{i}'].values.reshape(-1, 1)
            )

        X.drop(columns=['first_page', 'first_page_split', 'first_page_0_5'], inplace=True)

        return X
