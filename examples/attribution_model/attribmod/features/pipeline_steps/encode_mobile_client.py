import os
import pickle

import numpy as np
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeMobileClient(TransformerMixin, BaseEstimator):
    def __init__(self, encoder=None, missing=None):
        self.encoder = encoder
        self.missing = missing

    def fit(self, X, y=None):
        return self

    def _provide_fitted_encoder(self, X):
        if self.encoder is None:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(
                np.array([self.missing] + X['wtr_mobile_client'].unique().tolist()).reshape(-1, 1)
            )
        else:
            self.encoder = pickle.loads(self.encoder)
            # Set unknown values in encoder-categories to missing
            X.loc[
                ~X['wtr_mobile_client'].isin(self.encoder.categories_[0]), 'wtr_mobile_client'
            ] = self.missing

    def _agg_mob_client(self, text: str):
        if 'iPhone' in text:
            return 'iPhone'
        if 'iPad' in text:
            return 'iPad'
        if 'Apple Macintosh' in text:
            return 'Apple_Mac'
        if 'Huawei MediaPad' in text:
            return 'Huawei_Tablet'
        if 'Huawei' in text and 'lite' in text.lower():
            return 'Huawei_lite'
        if 'Huawei' in text and 'pro' in text.lower():
            return 'Huawei_Pro'
        if 'Huawei Mate' in text:
            return 'Huawei_Mate'
        if 'Huawei' in text:
            return 'Huawei'
        if 'Samsung Galaxy Tab' in text:
            return 'Samsung_Tab'
        if 'Samsung Galaxy S' in text:
            return 'Samsung_S'
        if 'Samsung' in text:
            return 'Samsung'
        if 'Sony' in text:
            return 'Sony'
        if 'Xiaomi' in text:
            return 'Xiaomi'
        if 'Motorola' in text:
            return 'Motorola'
        if 'LG' in text:
            return 'LG'
        if text in [
            'any Desktop & Laptop',
            'Diverse PC&Laptop',
            'Lenovo Windows Desktop',
            'Acer Windows Desktop',
            'Dell Windows Desktop',
            'Medion Windows Desktop',
            'Asus Windows Desktop',
            'Toshiba Windows Desktop',
        ]:
            return 'DesktopLaptop'
        if 'Other Android' in text or 'any Android' in text:
            return 'Android'
        if 'Google Pixel' in text or 'HMD' in text or 'HTC' in text or 'Oneplus' in text:
            return 'Android'
        if 'Lenovo' in text:
            return 'Lenovo'
        if 'Windows' in text and 'Touch' in text:
            return 'Windows_Touch'
        if 'Amazon' in text or 'ZenPad' in text or 'Tablet' in text or 'Vodafone TAB' in text:
            return 'Android_Tablet'
        if text == 'Unknown':
            return 'Unknown'
        return 'others'

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        # Reduce device count
        X['wtr_mobile_client'] = X['wtr_mobile_client'].apply(self._agg_mob_client)
        self._provide_fitted_encoder(X)
        if self.encoder.inverse_transform([[0]]) != self.missing:
            raise ValueError(
                'OrdinalEncoder mapped 0 nicht auf missing, sondern auf %s'
                % self.encoder.inverse_transform([[0]])[0][0]
            )
        X['wtr_mobile_client'] = self.encoder.transform(
            X['wtr_mobile_client'].values.reshape(-1, 1)
        )
        return X
