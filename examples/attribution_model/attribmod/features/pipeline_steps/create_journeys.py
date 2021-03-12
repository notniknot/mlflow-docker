import os

import pandas as pd
from attribmod.utils import my_logging
from sklearn.base import BaseEstimator, TransformerMixin

LOGGER = my_logging.logger(os.path.basename(__file__))


class CreateJourneys(TransformerMixin, BaseEstimator):
    def __init__(self, keep_cols=None):
        self.keep_cols = keep_cols

    def fit(self, X, y=None):
        return self

    def _pivot(self, X: pd.DataFrame) -> pd.DataFrame:
        base_cj = pd.pivot_table(
            X[X.no < 10], index='journey_id', columns='no', values=self.keep_cols, aggfunc='max'
        )

        # Vernuenftige Spaltenbenennung
        base_cj.columns = [f'{col[0]}_{col[1]}' for col in list(base_cj.columns)]

        # Nan zu None fuer touchpoint_X
        base_cj = base_cj.where(pd.notnull(base_cj), None)

        # Index zu Spalte
        base_cj.reset_index(inplace=True)

        # Fehlende Spalten mit Standardwert initialisieren
        max_col_no = X.no.max()
        if max_col_no < 9:
            for i in range(max_col_no + 1, 10):
                # Assign new columns
                missing_cols = {f'{col}_{i}': 0 for col in self.keep_cols}
                base_cj = base_cj.assign(**missing_cols)

        # Differenz zwischen dem Datum der allerletzten Session und dem Datum der letzten 9 Sessions (ohne die allerletzte Session)
        # in Stunden
        for i in range(1, 10):
            base_cj[f'wtr_vt_diff_{i}'] = 10 * 365 * 24  # 10 Jahre in Stunden

            # Keep 10 years as default value
            if i > max_col_no:
                base_cj[f'wtr_vt_diff_{i}'] = base_cj[f'wtr_vt_diff_{i}'].astype('float64')
                continue

            with_time = ~base_cj[f'wtr_visit_time_start_{i}'].isna()
            base_cj.loc[with_time, f'wtr_vt_diff_{i}'] = (
                base_cj.loc[with_time, 'wtr_visit_time_start_0']
                - base_cj.loc[with_time, f'wtr_visit_time_start_{i}']
            ).astype('timedelta64[h]')

        base_cj.drop(
            columns=[col for col in base_cj.columns if col.startswith('wtr_visit_time_start_')],
            inplace=True,
        )

        base_cj.fillna(0, inplace=True)
        return base_cj

    def _get_session_count(self, X: pd.DataFrame, base_cj: pd.DataFrame) -> pd.DataFrame:
        n_sessions = (
            X[['journey_id', 'sid']]
            .groupby('journey_id')
            .count()
            .reset_index()
            .rename({'sid': 'nsessions'}, axis=1)
        )
        return base_cj.merge(n_sessions, how='inner', on='journey_id')

    def _known_visitor(self, base_cj: pd.DataFrame) -> pd.DataFrame:
        base_cj.loc[:, 'web_partnerid'] = base_cj.loc[:, 'web_partnerid_1':'web_partnerid_9'].max(1)
        return base_cj

    def _sort_columns_alphabetically(self, base_cj: pd.DataFrame) -> pd.DataFrame:
        cols = base_cj.columns.to_list()
        cols = sorted(cols)
        base_cj = base_cj[cols]
        return base_cj

    def _antrag(self, X: pd.DataFrame, base_cj: pd.DataFrame) -> pd.DataFrame:
        antrag = X[X.journey_end == True][['journey_id', 'antrag']]
        base_cj = base_cj.merge(antrag, how='inner', on='journey_id')
        return base_cj

    def transform(self, X):
        LOGGER.debug(f'Pipeline step {type(self).__name__}')
        base_cj = self._pivot(X)
        base_cj = self._get_session_count(X, base_cj)
        base_cj = self._known_visitor(base_cj)
        base_cj = self._sort_columns_alphabetically(base_cj)
        base_cj = self._antrag(X, base_cj)
        return base_cj
