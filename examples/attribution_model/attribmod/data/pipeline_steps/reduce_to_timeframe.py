import os

import pandas as pd
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class ReduceToTimeframe(TransformerMixin, BaseEstimator):
    def __init__(self, from_date=None, to_date=None):
        self.from_date = from_date
        self.to_date = to_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        journey_end = X.loc[X['journey_end'] == True, ['journey_id', 'zeitpunkt']]
        X = X.merge(journey_end, on='journey_id', how='left', suffixes=['', '_last'])
        # We add 1 day to to_date because pandas keeps the time as 00:00:00 (Midnight)
        X = X.loc[
            # e.g. 'zeitpunkt_last' >= 2020-01-01 00:00:00
            (X['zeitpunkt_last'] >= pd.to_datetime(self.from_date))
            # e.g. 'zeitpunkt_last' < 2020-01-05 00:00:00
            & (X['zeitpunkt_last'] < pd.to_datetime(self.to_date) + pd.Timedelta("1 days"))
            & ((X['zeitpunkt_last'] - X['zeitpunkt']).dt.days <= 365)
        ]
        X = X.drop(columns='zeitpunkt_last')
        return X
