import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class CreateDateTimeExtracts(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X['wtr_visit_time_month'] = X['wtr_visit_time_start'].dt.month
        X['wtr_visit_time_dayofweek'] = X['wtr_visit_time_start'].dt.dayofweek
        X['wtr_visit_time_hour'] = X['wtr_visit_time_start'].dt.hour
        return X
