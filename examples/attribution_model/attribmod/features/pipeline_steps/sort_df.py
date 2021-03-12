import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class SortDF(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.shape = X.shape

        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        return X.sort_values(
            ['journey_id', 'wtr_visit_time_start'], ascending=[True, False]
        ).reset_index(drop=True)
