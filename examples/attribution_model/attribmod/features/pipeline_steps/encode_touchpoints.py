import os
import pickle

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeTouchpoints(TransformerMixin, BaseEstimator):
    def __init__(self, encoder=None, missing=None):
        self.encoder = encoder
        self.missing = missing

    def fit(self, X, y=None):
        return self

    def _provide_fitted_encoder(self, X):
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit([self.missing] + X['touchpoint'].unique().tolist())
        else:
            self.encoder = pickle.loads(self.encoder)

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        self._provide_fitted_encoder(X)
        if self.encoder.inverse_transform([0]) != [self.missing]:
            raise ValueError(
                'LabelEncoder mapped 0 nicht auf missing, sondern auf %s'
                % self.encoder.inverse_transform([0])[0]
            )
        X['touchpoint'] = self.encoder.transform(X['touchpoint'])
        return X
