import os
import pickle
import re

import numpy as np
from attribmod.utils import my_logging
from packaging import version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

LOGGER = my_logging.logger(os.path.basename(__file__))


class EncodeOS(TransformerMixin, BaseEstimator):
    def __init__(self, encoder=None, missing=None):
        self.encoder = encoder
        self.missing = missing

    def fit(self, X, y=None):
        return self

    def _provide_fitted_encoder(self, X):
        if self.encoder is None:
            self.encoder = OrdinalEncoder()
            self.encoder.fit(
                np.array([self.missing] + X['wtr_os'].unique().tolist()).reshape(-1, 1)
            )
        else:
            self.encoder = pickle.loads(self.encoder)
            # Set unknown values in encoder-categories to missing
            X.loc[~X['wtr_os'].isin(self.encoder.categories_[0]), 'wtr_os'] = self.missing

    def _create_os_mapper(self, oslist: list, starter):
        if isinstance(starter, str):
            starter = [starter]

        oslist_new = dict()
        for starter_str in starter:
            oslist_new = {
                **oslist_new,
                **{
                    osname: re.match(r".* (\d+(?:\.\d+)?).*", osname)[1]
                    for osname in oslist
                    if osname.startswith(starter_str) and re.match(r".* (\d+(?:\.\d+)?).*", osname)
                },
            }
        oslist_new = {
            k: v for k, v in sorted(oslist_new.items(), key=lambda item: version.parse(item[1]))
        }
        prefix = starter[0]
        prefix = prefix.replace(' ', '_')
        oslist_new = {
            osname: prefix + 'neuer'
            if key > len(oslist_new) * 0.8 - 1
            else prefix + 'aelter'
            if key > len(oslist_new) * 0.4 - 1
            else prefix + 'alt'
            for key, osname in enumerate(oslist_new)
        }
        return oslist_new

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        # Create mapper
        a_os_mapper = self._create_os_mapper(X['wtr_os'].unique(), 'Android ')
        ios_mapper = self._create_os_mapper(X['wtr_os'].unique(), 'iOS ')
        macos_mapper = self._create_os_mapper(
            X['wtr_os'].unique(),
            ['Mac OS X ', 'Apple Mac OS X ', 'macOS ', 'Apple macOS ', 'Apple OS X '],
        )
        win_mapper = {
            'Windows 95': 'Windows_alt',
            'Windows 98': 'Windows_alt',
            'Windows 8': 'Windows_aelter',
            'Windows 8.1': 'Windows_aelter',
        }
        winphone_mapper = self._create_os_mapper(X['wtr_os'].unique(), 'Windows Phone ')
        winserver_mapper = {
            'Windows NT': 'Windows_NTServer',
            'Windows Server 2003': 'Windows_NTServer',
            'Windows 2000': 'Windows_NTServer',
        }
        sonstiges_oft_mapper = {
            'Mac': 'Mac',
            'Windows Vista': 'Windows_Vista',
            'Windows XP': 'Windows_XP',
            'Unix/Linux': 'LUnix',
            'null': 'null',
            'Windows 10': 'Windows_10',
            'Windows 7': 'Windows_7',
        }
        mapper = {
            **a_os_mapper,
            **ios_mapper,
            **macos_mapper,
            **win_mapper,
            **winphone_mapper,
            **winserver_mapper,
            **sonstiges_oft_mapper,
        }

        # Apply mapping to reduce os count
        X['wtr_os'] = X['wtr_os'].apply(lambda text: mapper.get(text, 'sonstiges'))
        self._provide_fitted_encoder(X)
        # Encode mapped values
        if self.encoder.inverse_transform([[0]]) != self.missing:
            raise ValueError(
                'OrdinalEncoder mapped 0 nicht auf missing, sondern auf %s'
                % self.encoder.inverse_transform([0])[0][0]
            )
        X['wtr_os'] = self.encoder.transform(X['wtr_os'].values.reshape(-1, 1))
        return X
