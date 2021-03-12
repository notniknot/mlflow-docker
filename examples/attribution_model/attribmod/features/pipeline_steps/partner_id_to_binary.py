import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class PartnerIdToBinary(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X['web_partnerid'] = X['web_partnerid'].astype('int8')
        return X
