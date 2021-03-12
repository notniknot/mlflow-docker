import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class GenerateJourneyId(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X.rename(columns={'wtr_visit_time_end': 'zeitpunkt'}, inplace=True)
        X.sort_values(by=['user_id', 'zeitpunkt'], inplace=True)
        df_antrag = X['antrag'] == 1
        df_open = X.shift(-1)['user_id'] != X['user_id']
        X['journey_end'] = df_antrag | df_open
        X['journey_beg'] = X.shift(1)['journey_end'].fillna(True).astype(int)
        X['journey_id'] = X['journey_beg'].cumsum()
        return X
