import os
from functools import lru_cache
from typing import Tuple
import gzip
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from attribmod.data import abt_data
from attribmod.models import attrib_encoder
from attribmod.utils import datafilefinder


@lru_cache()
def _get_tp_encoder() -> LabelEncoder:
    encoder = attrib_encoder.get_encoder()
    return encoder['touchpoint']


@lru_cache()
def _get_tps_cnt() -> dict:
    if 'MLFLOW_ARTIFACT_FSC' in os.environ:
        filepath = os.getenv('MLFLOW_ARTIFACT_FSC')
    else:
        filepath = datafilefinder.get_path('models', 'fsc.pickle.gz')
    with gzip.open(filepath, mode='rb') as file:
        tps_cnt = pickle.load(file)
    return tps_cnt


@lru_cache()
def _get_n_classes() -> int:
    return len(_get_tp_encoder().classes_)


def _get_channel_weights(channel_1, channel_2) -> Tuple[float, float]:
    """Liefert die Gewichtung zwei Kanäle zurueck.

    Hierbei ist zu beachten, dass in der ABT die Reihenfolge der Touchpoints umgekehrt ist.
    Entsprechend ist die tatsächliche Reihenfolge channel_2 -> channel_1

    Args:
        channel_1 (int): Erster Kanal (gemaess umgekehrter Reihenfolge)
        channel_2 (int): Zweiter Kanal (gemaess umgekehrter Reihenfolge)

    Returns:
        Tuple(float, float): Gewichtung der beiden Kanäle
    """
    if isinstance(channel_1, str):
        channel_1 = _get_tp_encoder().transform([channel_1])[0]
    if isinstance(channel_2, str):
        channel_2 = _get_tp_encoder().transform([channel_2])[0]
    weight = _get_tps_cnt().get((channel_1, channel_2), 0)
    if weight == 0:
        return 1, 1
    if weight == 1:
        weight = 0.999
    return 1 / (1 - weight), 1 / weight


def _compute_channel_weights(row_start: int, row: pd.Series) -> list:
    """Berechne die Gewichte für zwei aufeinanderfolgende Spalten

    Args:
        row_start (int): Die erste Spalte
        row (pd.Series): Eine Zeile mit den Touchpoints

    Returns:
        list: Eine Liste mit einem Gewicht für jeden Kanal
    """
    result = [0] * _get_n_classes()
    if row['touchpoint_%i' % row_start] == 0 or row['touchpoint_%i' % (row_start + 1)] == 0:
        return result
    weights = _get_channel_weights(
        row['touchpoint_%i' % row_start], row['touchpoint_%i' % (row_start + 1)]
    )
    result[row['touchpoint_%i' % row_start]] = weights[0]
    result[row['touchpoint_%i' % (row_start + 1)]] = weights[1]
    return result


def apply_fsc(df: pd.DataFrame, proba_df: pd.DataFrame) -> pd.DataFrame:
    """Korrigiert das Attributionsmodell nach einem einfachen Vorgehen namens Frequency of Successor Correction

    Args:
        df (pd.DataFrame): Ein DataFrame analog zur Analytical Base Table
        proba_df (pd.DataFrame): Das Ergebnis des Attributionsmodells angewendet auf df

    Returns:
        pd.DataFrame: proba_df mit angewandter Korrektur
    """
    mask = df.touchpoint_1 > 0
    if mask.sum() == 0:
        return proba_df

    tps = df.loc[mask, ['touchpoint_%i' % i for i in range(10)]]
    tps = tps.astype('int8')

    # Berechne für jedes aufeinanderfolgende Kanalpaar das Gewicht und speichere es für die Kanäle ab
    weights = None
    for i in range(9):
        weights_tmp = tps.apply(lambda row: _compute_channel_weights(i, row), axis=1)
        weights_tmp = np.stack(weights_tmp, axis=0)
        if weights is None:
            weights = weights_tmp
        else:
            weights += weights_tmp

    n_classes = _get_n_classes()
    # Multipliziere die Kanalverteilung mit den Gewichten
    proba_df.loc[mask, 'channel_1' : 'channel_%i' % (n_classes - 1)] = proba_df.loc[
        mask, 'channel_1' : 'channel_%i' % (n_classes - 1)
    ].multiply(weights[:, 1:])
    # Berechne die neue Summe der Kanalverteilung pro CJ
    proba_df['summe_weighted'] = proba_df.loc[:, 'channel_1' : 'channel_%i' % (n_classes - 1)].sum(
        1
    )
    # Normiere die Kanalverteilung, so dass die Summe wieder mit der alten Summe übereinstimmt
    proba_df.loc[mask, 'channel_1' : 'channel_%i' % (n_classes - 1)] = (
        proba_df.loc[mask, 'channel_1' : 'channel_%i' % (n_classes - 1)]
        .div(proba_df.loc[mask, 'summe_weighted'], axis=0)
        .multiply(proba_df.loc[mask, 'summe'], axis=0)
    )

    return proba_df


def fit_fsc() -> None:
    """Ermittelt die grundlegenden statistischen Werte anhand der Analytical Base Table, um
    die Frequency of Successor Correction anwenden zu können.
    """
    abt = abt_data.get_abt()
    abt = abt[abt.antrag == 1]

    # Es werden nur die Touchpoints benoetigt
    tps = abt[['touchpoint_%i' % i for i in range(10)]]
    tps = tps.astype('int8')

    # Alle Kanal-Paare ermitteln und zaehlen, wie haeufig sie vorkommen
    tps_cnt = dict()
    for i in range(9):
        tps_tmp = tps[tps['touchpoint_%i' % (i + 1)] > 0]
        for row in tps_tmp.iterrows():
            element = (row[1]['touchpoint_%i' % i], row[1]['touchpoint_%i' % (i + 1)])
            if element[0] != element[1]:
                if element in tps_cnt:
                    tps_cnt[element] += 1
                else:
                    tps_cnt[element] = 1

    # Nur Kombinationen, die mindestens 10x vorkommen
    tps_cnt = {key: value for key, value in tps_cnt.items() if value >= 10}

    # Pro Kanal berechnen, wie haeufig er als erstes vorkommt (hier dann als zweites, da die Reihenfolge vertauscht ist)
    tp_encoder = _get_tp_encoder()
    sum_channel = dict()
    for i in range(1, len(tp_encoder.classes_)):
        sum_channel[i] = sum([tps_cnt[el] for el in tps_cnt if el[1] == i])

    # Mit obiger Summe normieren
    for el in tps_cnt:
        tps_cnt[el] = tps_cnt[el] / sum_channel[el[1]]

    pickle_path = Path(__file__).resolve().parent.parent.parent / 'models' / 'fsc.pickle.gz'
    with gzip.open(pickle_path, mode='wb') as file:
        pickle.dump(tps_cnt, file)


if __name__ == '__main__':
    _get_tp_encoder()
