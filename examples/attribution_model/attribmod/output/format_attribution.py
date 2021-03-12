import os
import re

import importlib_metadata as metadata
import numpy as np
import pandas as pd
from attribmod.models.attrib_encoder import get_encoder
from attribmod.output.write_to_table import write_to_table
from attribmod.utils import my_logging
from attribmod.utils.datafilefinder import get_path

LOGGER = my_logging.logger(os.path.basename(__file__))


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to specified format.

    Args:
        df (pd.DataFrame): DataFrame with the columns to rename

    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    df.rename({'web_order_id': 'ATT_BESTELLNUMMER', 'sid': 'ATT_SESSION_ID'}, axis=1, inplace=True)
    enc = get_encoder()
    tp_enc = enc['touchpoint']
    renames = {}
    converts = {}
    for column in df.columns:
        if not column.startswith('channel_'):
            continue
        converts[column] = np.float64
        if column.endswith('_0'):
            renames[column] = 'ATT_NICHT_ZUORDENBARE_EFFEKTE'
        else:
            raw_name = tp_enc.inverse_transform([int(column[8:])])[0]
            renames[column] = 'ATT_' + re.sub('[^a-zA-Z0-9\n\.]', '_', raw_name).upper()
    df = df.astype(converts)
    df.rename(renames, axis=1, inplace=True)
    return df


def _format_df(result_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Format given DataFrame to specified format.

    Args:
        df (pd.DataFrame): DataFrame to format
        mapping_df (pd.DataFrame): DataFrame for mapping purposes with journey_id and web_order_id

    Returns:
        pd.DataFrame: Formatted DataFrame
    """
    df = result_df.merge(mapping_df, how='inner', on='journey_id')
    df.drop(['pred_proba', 'summe', 'summe_weighted', 'journey_id'], axis=1, inplace=True)
    df = _rename_columns(df)
    df['ATT_UPLOAD_TIME'] = pd.to_datetime(np.nan)
    df['ATT_PREDICTION_TIME'] = pd.to_datetime('today')
    df['ATT_MODEL_VERSION'] = metadata.version('attribmod')
    cols = df.columns.to_list()
    cols.remove('ATT_BESTELLNUMMER')
    cols.remove('ATT_SESSION_ID')
    cols.remove('ATT_UPLOAD_TIME')
    cols.remove('ATT_PREDICTION_TIME')
    cols.remove('ATT_MODEL_VERSION')
    cols.remove('ATT_NICHT_ZUORDENBARE_EFFEKTE')
    cols = (
        [
            'ATT_BESTELLNUMMER',
            'ATT_SESSION_ID',
            'ATT_PREDICTION_TIME',
            'ATT_UPLOAD_TIME',
            'ATT_MODEL_VERSION',
        ]
        + cols
        + ['ATT_NICHT_ZUORDENBARE_EFFEKTE']
    )
    df = df.reindex(cols, axis=1)
    return df


def orchestrate_output_handling():
    """
    Orchestrate model application steps.
    """
    LOGGER.info('Starting orchestration of output handling')

    # * Load attribution
    LOGGER.info('Loading attribution.feather')
    attribution_file_path = get_path('data/processed', 'attribution.feather')
    attribution_df = pd.read_feather(attribution_file_path)

    # * Load mapping
    LOGGER.info('Loading orderid_mapping.feather')
    mapping_file_path = get_path('data/processed', 'orderid_mapping.feather')
    mapping_df = pd.read_feather(mapping_file_path)

    # * Merge and format table
    LOGGER.info('Format attribution table')
    df = _format_df(attribution_df, mapping_df)

    # * Store table
    LOGGER.info('Writing formatted attribution table to database')
    write_to_table(df)
    LOGGER.info('Finished with output handling')
