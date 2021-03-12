import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class DefineAntrag(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X['web_basket_product'] = (X['web_basket_product'].notnull()).astype(int)
        X.rename({'web_basket_product': 'antrag'}, axis=1, inplace=True)
        return X
