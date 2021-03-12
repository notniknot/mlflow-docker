import os

from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeBrowser(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        X['wtr_browser_int'] = 1 * X['wtr_browser'].str.contains('Google Chrome').astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 2 * X['wtr_browser'].str.contains(
            'Firefox'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 3 * X['wtr_browser'].str.contains(
            'Internet Explorer'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 4 * X['wtr_browser'].str.contains(
            'Microsoft Edge'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 5 * X['wtr_browser'].str.contains(
            'Samsung Browser'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 6 * X['wtr_browser'].str.contains(
            'Native App'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 7 * X['wtr_browser'].str.contains(
            'Chrome for iOS'
        ).astype('int8')
        X['wtr_browser_int'] = X['wtr_browser_int'] + 8 * X['wtr_browser'].str.contains(
            'Safari'
        ).astype('int8')
        return X
