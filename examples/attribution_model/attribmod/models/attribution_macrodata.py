import pandas as pd
from attribmod.models import convmod_classifier, attrib_encoder


def apply_macrodata(df: pd.DataFrame, proba_df: pd.DataFrame) -> pd.DataFrame:
    """Wendet den Makrodaten-Ansatz auf die Daten an.

    Über attribution.model nutzen, nicht direkt

    Args:
        df: DataFrame basieren auf der Analytical Base Table
        proba_df: Ergebnis aus dem Attribution Model

    Returns:
        Angepasstes proba_df
    """

    clf = convmod_classifier.get_classifier()
    features = clf.get_booster().feature_names
    # Die Spalten, die im CLassifier nicht vorgesehen sind, speichern, um sie später zu droppen
    columns_drop = [col for col in df.columns.tolist() if col not in features]

    # Encoder für die Touchpoints holen
    le = attrib_encoder.get_encoder()['touchpoint']
    # Nur die CJ, für die auch eine Gewichtung vorliegt
    df2 = df[df.index.isin(proba_df.index)].copy()

    # Channel-Liste
    channels = ['Display', 'Affiliate', 'Partner', 'Social', 'Online Video', 'SEA NonBrand']
    for channel in channels:
        # Channel-bezogenen Spalten ermitteln
        columns = [col for col in df.columns.tolist() if col.startswith('macrodata_%s' % channel.lower().replace(' ', ''))]
        # Die Channel-bezogenen Daten auf 0 setzen
        df3 = df2.copy()
        df3.loc[:, columns] = 0
        # Aktuellen Channel speichern
        channel_i = le.transform([channel])[0]
        # Was wäre die Prediction wenn die Information zum Kanal fehlen würde
        pred = clf.predict_proba(df3.drop(columns=columns_drop))[:, 1]
        # Beide Predictions von einander abziehen (und bei 0 nach unten abschneiden)
        proba_df['channel_%i' % channel_i] += (proba_df.pred_proba - pred).apply(lambda value: value if value >= 0 else 0)

    return proba_df
