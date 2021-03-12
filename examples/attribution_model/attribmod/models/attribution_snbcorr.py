"""Korrektur von SEA Non Brand

"""

import pandas as pd
from attribmod.models import attrib_encoder


def apply_snbcorr(proba_df: pd.DataFrame) -> pd.DataFrame:

    factor = 2.3

    le = attrib_encoder.get_encoder()['touchpoint']
    sea_nb = le.transform(['SEA NonBrand'])[0]
    channel_no = len(le.classes_) - 1
    proba_df2 = proba_df.copy()

    proba_df2['channel_%i' % sea_nb] = proba_df2['channel_%i' % sea_nb]*factor
    mask = proba_df2.loc[:, 'channel_1':'channel_%i' % channel_no].sum(1) > 1
    proba_df2.loc[mask, 'channel_1':'channel_%i' % channel_no] = proba_df2.loc[mask, 'channel_1':'channel_%i' % channel_no].div(proba_df2.loc[mask, 'channel_1':'channel_%i' % channel_no].sum(1), axis=0)

    return proba_df2
