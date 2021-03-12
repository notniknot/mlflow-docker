import pandas as pd
from attribmod.models import convmod_classifier, attrib_encoder, attribution_fsc, attribution_macrodata, attribution_snbcorr


def _col_list(prefix: str, start: int, end: int) -> list:
    return [f'{prefix}{i}' for i in range(start, end)]


def model(df: pd.DataFrame, all_cj: bool = True, use_macrodata: bool = False, use_fsc: bool = False, use_snbcorr: bool = False) -> pd.DataFrame:
    """Führt die eigentliche Attribution durch

    Args:
        df (pd.DataFrame): Dataframe mit allen Customer Journeys, analog zur Analytical Base Table
        all_cj (pd.DataFrame): Sollen alle CJ durch das Attributionsmodell laufen (Default) oder nur die,
                               die gemäß Conversion Model eine Antragswahrscheinlichkeit >= 50% haben
        use_macrodata (bool): Soll der Macrodata-Ansatz angewandt werden oder nicht (Default)
        use_fsc (bool): Soll FSC angewandt werden oder nicht (Default)
        use_snbcorr (bool): Korrektur SEA Non Brand

    Returns:
        pd.DataFrame: Die Attribution
    """

    df = df.copy()
    le = attrib_encoder.get_encoder()['touchpoint']
    channels_no = len(le.classes_) - 1

    xgbc = convmod_classifier.get_classifier()
    df['pred_proba'] = xgbc.predict_proba(df.drop(columns='antrag'))[:, 1]
    if not all_cj:
        df = df[df.pred_proba >= xgbc.threshold]

    prefixes = [col[:-1] for col in df.columns if col.endswith('_9')]

    # Für jede CJ berechnen, ob ein Kanal beteiligt ist oder nicht
    channel_involved = None
    for channel in range(1, len(le.classes_)):
        truefalse = pd.DataFrame({f'channel_{channel}': (df.loc[:, 'touchpoint_0':'touchpoint_9'] == channel).any(1)},
                                 index=df.index)
        if channel_involved is None:
            channel_involved = truefalse
        else:
            channel_involved = channel_involved.join(truefalse, how="outer")

    # Zentraler Abschnitt. Entferne einen Kanal nach dem anderen und berechne die neue Antragswahrscheinlichkeit
    pred_proba_all = dict()

    for channel in range(1, len(le.classes_)):

        # print(f'Kanal: {le.inverse_transform([channel])[0]}')

        # Lösche den Kanal channel und verschiebe die anderen Einträge entsprechend
        df2 = df.drop(columns='pred_proba').copy()
        df2 = df2.loc[channel_involved['channel_%i' % channel], :]
        # Alle Sessions durchgehen
        for i in reversed(range(10)):
            # Alle Session-bezogenen Spalten durchgehen
            for prefix in prefixes:
                # Verschieben der Einträge beim letzten Eintrag nicht notwendig
                if i < 9:
                    df2.loc[df[f'touchpoint_{i}'] == channel, _col_list(prefix, i, 9)] = df2.loc[df[f'touchpoint_{i}'] == channel, _col_list(prefix, i + 1, 10)].rename(columns=dict(zip(_col_list(prefix, i + 1, 10), _col_list(prefix, i, 9))))
                # Die erste (9.) Session mit missing-Werten auffüllen
                if prefix == 'wtr_vt_diff_':
                    value = df2.wtr_vt_diff_9.max()
                else:
                    value = 0
                df2.loc[df[f'touchpoint_{i}'] == channel, f'{prefix}9'] = value

        df2_missing = df2[df2.touchpoint_0 == 0].copy()
        df2 = df2[df2.touchpoint_0 != 0]

        # Unerwünschte Spalten auf 0 setzen (Information Leakage)
        df2['web_partnerid_0'] = 0
        df2['wtr_avg_page_duration_0'] = 0
        df2['wtr_vt_diff_0'] = 0

        # Berechne Antragswahrscheinlichkeit
        result_df = pd.DataFrame({f'channel_{channel}': xgbc.predict_proba(df2.drop(columns='antrag'))[:, 1]},
                                 index=df2.index)

        # Die CJ, die nur aus einem Kanal bestehen, noch dran hängen
        pred_proba_all[channel] = result_df.append(pd.DataFrame({f'channel_{channel}': [0.0]*df2_missing.shape[0]},
                                                                index=df2_missing.index))
        # print(pred_proba_all[channel])

    # Alle Antragswahrscheinlichkeiten joinen
    proba_df = None

    for channel, df2 in pred_proba_all.items():
        if proba_df is None:
            proba_df = df2
        else:
            proba_df = proba_df.join(df2, how="outer")

    # Ist eine Zelle nicht gefüllt, bestand die CJ ausschließlich aus diesem einen Kanal.
    # Daher wird hier 0 gesetzt, als niedrigst mögliche Wahrscheinlichkeit.
    proba_df = proba_df.fillna(0)

    proba_df = proba_df.merge(df[['pred_proba']], how='left', left_on='journey_id', right_on='journey_id')

    # Berechnen, welche Kanäle in einer Customer Journey involviert sind.
    proba_df = proba_df.join(channel_involved, how='inner', rsuffix='_inv')
    for channel in range(1, len(le.classes_)):
        proba_df.loc[proba_df[f'channel_{channel}_inv'] == False, f'channel_{channel}'] = 1

    proba_df.drop(columns=[col for col in proba_df.columns if col.endswith('_inv')], inplace=True)

    # Auf alle Wahrscheinlichkeiten 1-p anwenden, um die stärksten Auswirkungen den größten Wert zuzuordnen.

    proba_df.loc[:, 'channel_1':'channel_%i' % channels_no] = -1 * proba_df.loc[:, 'channel_1':'channel_%i' % channels_no].subtract(proba_df.pred_proba, axis=0)
    proba_df = proba_df.applymap(lambda val: val if val >= 0 else 0)
    proba_df['summe'] = proba_df.loc[:, 'channel_1':'channel_%i' % channels_no].sum(axis=1)
    # Wenn die Summe kleiner als die ursprüngliche Wahrscheinlichkeit ist, werden die Einzel-Anteile entsprechend aufgewertet
    proba_df.loc[proba_df.summe<proba_df.pred_proba, 'channel_1':'channel_%i' % channels_no] = proba_df.loc[proba_df.summe<proba_df.pred_proba, 'channel_1':'channel_%i' % channels_no].mul(proba_df.pred_proba, axis=0).div(proba_df.summe, axis=0)
    proba_df['summe'] = proba_df.loc[:, 'channel_1':'channel_%i' % channels_no].sum(axis=1)
    # proba_df = proba_df[proba_df.summe > 0]

    # Nur die CJ, die auch im Attribution Model beachtet wurden
    df3 = df[df.index.isin(proba_df.index)].drop(columns='pred_proba').copy()
    
    if use_fsc:
        proba_df = attribution_fsc.apply_fsc(df3, proba_df)

    if use_macrodata:
        proba_df = attribution_macrodata.apply_macrodata(df3, proba_df)

    if use_snbcorr:
        proba_df = attribution_snbcorr.apply_snbcorr(proba_df)

    # Summe nicht größer 1, aber größer Threshold
    proba_df['summe'] = proba_df.loc[:, 'channel_1': 'channel_%i' % channels_no].sum(axis=1)
    proba_df['summe2'] = proba_df.summe*xgbc.threshold + (1-xgbc.threshold)
    proba_df.loc[:, 'channel_1':'channel_%i' % channels_no] = proba_df.loc[:, 'channel_1':'channel_%i' % channels_no].div(proba_df.summe, axis=0).mul(proba_df.summe2, axis=0)
    gr1 = proba_df.summe2 > 1
    proba_df.loc[gr1, 'channel_1': 'channel_%i' % channels_no] = proba_df.loc[gr1, 'channel_1': 'channel_%i' % channels_no].div(proba_df.summe2, axis=0)
    proba_df.loc[:, 'channel_1': 'channel_%i' % channels_no] = proba_df.loc[:, 'channel_1': 'channel_%i' % channels_no].fillna(0)
    proba_df['summe'] = proba_df.loc[:, 'channel_1': 'channel_%i' % channels_no].sum(axis=1)
    proba_df['channel_0'] = 1 - proba_df['summe']
    proba_df.drop(columns='summe2', inplace=True)

    # Spaltenreihenfolge korrigieren, da channel_0 ans Ende angehängt wurde
    cols1 = ['channel_%i' % i for i in range(channels_no + 1)]
    cols2 = [col for col in proba_df.columns if col not in cols1]
    proba_df = proba_df[cols1 + cols2]

    return proba_df


if __name__ == '__main__':
    from pathlib import Path
    filepath = Path('.').absolute().parent.parent / 'notebooks' / 'abt_idx_3031.feather'
    df = pd.read_feather(filepath).set_index('journey_id')
    print(model(df))
