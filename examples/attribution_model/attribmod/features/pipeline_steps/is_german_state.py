import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class IsGermanState(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        brd_laender = [
            'Nordrhein-westfalen',
            'Bayern',
            'Baden-wurttemberg',
            'Hessen',
            'Niedersachsen',
            'Rheinland-pfalz',
            'Berlin',
            'Hamburg',
            'Sachsen',
            'Saarland',
            'Schleswig-holstein',
            'Brandenburg',
            'Sachsen-anhalt',
            'Thuringen',
            'Bremen',
            'Mecklenburg-vorpommern',
        ]
        X['wtr_region'] = (X['wtr_region'].isin(brd_laender)).astype('int8')
        return X
