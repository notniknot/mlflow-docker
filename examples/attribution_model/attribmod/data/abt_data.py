import pandas as pd
from attribmod.utils.datafilefinder import get_path


def get_session() -> pd.DataFrame:
    """Liefert die Session-Daten zurück (pro Zeile eine Session)

    Returns:
        pd.DataFrame: Session-Daten
    """
    sessions_file = get_path('data/interim', 'base_sessions_cj.feather')
    return pd.read_feather(sessions_file)


def get_abt(avoid_information_leakage: bool = True, abt_file: str = None) -> pd.DataFrame:
    """Liefert die Analytical Base Table zurück

    Standardmäßig werden folgende Spalten auf 0 gesetzt, um Information Leakage zu vermeiden:
    web_partnerid_0, wtr_avg_page_duration_0, wtr_vt_diff_0, wtr_pages_in_session_0, bouncer_0

    Args:
        avoid_information_leakage (bool): Ob bestimmte Infos aus der letzten Session auf 0 gesetzt werden (Default) oder nicht
        abt_file (str): Übergabe eines eigenen Dateipfades zur Analytical Base Table

    Returns:
        pd.DataFrame: Analytical Base Table
    """

    if abt_file is None:
        abt_file = get_path('data/processed', 'analytic_base_table.feather')

    abt = pd.read_feather(abt_file)
    abt.set_index('journey_id', inplace=True)

    # Information Leakage
    if avoid_information_leakage:
        abt['web_partnerid_0'] = 0
        abt['wtr_avg_page_duration_0'] = 0
        abt['wtr_vt_diff_0'] = 0
        abt['wtr_pages_in_session_0'] = 0
        abt['bouncer_0'] = 0
        abt['produktseite_0'] = 0
        abt['content_depth_0'] = 0
        abt['number_products_0'] = 0

    cols = abt.columns.tolist()
    cols = sorted(cols)
    abt = abt[cols]

    return abt
