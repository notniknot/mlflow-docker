import gc
import os
import pickle

import pandas as pd
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

LOGGER = my_logging.logger(os.path.basename(__file__))


class CreateCampFields(TransformerMixin, BaseEstimator):
    def __init__(self, mlb=None, pca=None, pca_cols=None):
        self.mlb = mlb
        self.pca = pca
        self.pca_cols = pca_cols

    def fit(self, X, y=None):
        return self

    def _provide_fitted_binarizer(self, camp_split):
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(camp_split)
        else:
            self.mlb = pickle.loads(self.mlb)

    def _provide_fitted_pca(self, camp_split):
        if self.pca is None:
            self.pca = PCA(n_components=4)
            self.pca.fit(camp_split.sample(frac=0.2))
            self.pca_cols = camp_split.columns
        else:
            self.pca = pickle.loads(self.pca)
            self.pca_cols = pickle.loads(self.pca_cols)

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')

        # Kommt der Kunde über eine Kampagne oder nicht
        X['camp'] = (~X['wtm_kampagne'].isna()).astype('int8')

        # Zerlegen der Kampagnen in die Breite
        camp_split = (
            X['wtm_kampagne']
            .str.lower()
            .str.replace('non-', 'non')
            .str.split(pat='[ _-]')
            .fillna('missing')
        )
        X.drop(columns='wtm_kampagne', inplace=True)
        X_idx = X.index

        # OneHotEncoding mit MultiLabelBinarizer
        self._provide_fitted_binarizer(camp_split)
        camp_split = pd.DataFrame(
            self.mlb.transform(camp_split).astype('int8'), columns=self.mlb.classes_, index=X_idx
        )
        camp_split.columns = [f'camp_{col}' for col in camp_split.columns]

        self._provide_fitted_pca(camp_split)

        # Fehlende Spalten mit Standardwert initialisieren (if not trained)
        missing_cols = {col: 0 for col in camp_split.columns if col not in self.pca_cols}
        if missing_cols:
            camp_split = camp_split.assign(**missing_cols)
            camp_split = camp_split[self.pca_cols]

        camp_new = self.pca.transform(camp_split)
        pca_colnames = [f'camp_pca_{i}' for i in range(self.pca.n_components)]
        camp_new_df = pd.DataFrame(camp_new, columns=pca_colnames, index=X_idx)

        del camp_split
        del camp_new

        # Garbage Collector explizit aufrufen, da gleich der Speicher knapp wird
        gc.collect()

        X = X.join(camp_new_df)

        del camp_new_df

        return X
