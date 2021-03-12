from pathlib import Path
import pandas as pd


def _touchpoint_mapping(channel: str) -> str:
    """Wandelt einen Kanalnamen aus Webtrekk in einen Kanalnamen gemäß Attributionskanal um

    Args:
        channel (str): Kanalname aus Webtrekk

    Returns:
        str: Kanalname gemäß Attributionskana
    """

    if channel.startswith('Google') or channel.startswith('Bing') or channel.startswith('Yahoo'):
        return 'SEA'
    return channel.split(' ')[0]


def get_data() -> pd.DataFrame:
    """Liefert Visits und Anträge für 2020 aus Webtrekk zurück

    Returns:
        pd.DataFrame: Visits und Anträge für 2020 aus Webtrekk
    """
    data_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'external' / 'webtrekk_2020.xlsx'
    df = pd.read_excel(data_path).fillna(0)

    # Alle Spalten als Integer
    cols = [col for col in df.columns if col != 'kanal']
    df.loc[:, cols] = df.loc[:, cols].astype('int')

    # Kanäle wegwerfen, die wir im Attributionsmodell nicht betrachten
    df.kanal = df.kanal.replace('E-Mail LO', 'E-Mail_LO').replace('Online Video', 'Online_Video')
    # df = df[~df.kanal.isin(['E-Mail LO', 'mCD', 'Gesamt', 'mediacode_fehler'])].reset_index(drop=True)
    df = df[~df.kanal.isin(['Gesamt', 'mediacode_fehler'])].reset_index(drop=True)

    # Namensmapping
    df.kanal = df.kanal.apply(_touchpoint_mapping)
    df = df.groupby('kanal').sum().reset_index()
    df.kanal = df.kanal.str.replace('_', ' ')

    # SEA in SEA Brand und SEA Non Brand aufteilen
    sea_brand = df[df.kanal == 'SEA'].copy()
    sea_brand.kanal = 'SEA Brand'
    sea_brand.visits = sea_brand.visits_brand
    sea_brand.antraege = sea_brand.antraege_brand

    sea_nonbrand = df[df.kanal == 'SEA'].copy()
    sea_nonbrand.kanal = 'SEA NonBrand'
    sea_nonbrand.visits = sea_nonbrand.visits_nonbrand
    sea_nonbrand.antraege = sea_nonbrand.antraege_nonbrand

    df = df.append(sea_brand, ignore_index=True).append(sea_nonbrand, ignore_index=True)
    df = df[df.kanal != 'SEA']

    return df[['kanal', 'visits', 'antraege']]


if __name__ == '__main__':
    get_data()
