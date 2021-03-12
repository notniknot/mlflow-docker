import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class SetDtypes(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        return X.astype(
            {
                'first_page': 'category',
                'last_page': 'category',
                'wtr_campaigns_campaign': 'category',
            }
        )
