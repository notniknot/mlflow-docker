from attribmod.utils import dbconnection
import pandas as pd


def write_to_table(df: pd.DataFrame):
    """Write DataFrame with specified column types to table.

    Args:
        df (pd.DataFrame): DataFrame to write to table
    """
    da = dbconnection.DWH('data_analytics')
    df.to_sql('attribution_model_result', da.get_connection(), index=False, if_exists='append')
