import os
from pathlib import Path

import pandas as pd
from attribmod.data.pipeline_steps.assign_unknown_touchpoints import AssignUnknownTouchpoints
from attribmod.data.pipeline_steps.define_antrag import DefineAntrag
from attribmod.data.pipeline_steps.generate_journeyid import GenerateJourneyId
from attribmod.data.pipeline_steps.reduce_to_timeframe import ReduceToTimeframe
from attribmod.data.pipeline_steps.set_dtypes import SetDtypes
from attribmod.utils import my_logging
from attribmod.utils.datafilefinder import get_path
from sklearn.pipeline import Pipeline

LOGGER = my_logging.logger(os.path.basename(__file__))


def orchestrate_data_generation(df_or_path, from_date: str, to_date: str, train: bool = False):
    """Orchestrate data generation steps.

    Args:
        from_date (str, optional): Start date for sql-queries in "YYYY-MM-DD"-format. Defaults to None.
        to_date (str, optional): End date for sql-queries in "YYYY-MM-DD"-format. Defaults to None.
        train (bool, optional): Sets from_date to 1st Jan of last year if from_date is empty. Defaults to False.
    """
    LOGGER.info('Starting orchestration of data generation')

    if isinstance(df_or_path, pd.DataFrame):
        input_df = df_or_path
    else:
        LOGGER.info('Reading bs')
        input_df = pd.read_feather(df_or_path)

    data_gen_pipeline = Pipeline(
        steps=[
            ('dtypes', SetDtypes()),
            ('antrag', DefineAntrag()),
            ('journey_id', GenerateJourneyId()),
            ('unknown_touchpoints', AssignUnknownTouchpoints()),
            ('reduction', ReduceToTimeframe()),
        ]
    )
    params = {'reduction__from_date': from_date, 'reduction__to_date': to_date}
    data_gen_pipeline.set_params(**params)
    LOGGER.info('Executing pipeline')
    data_df = data_gen_pipeline.fit_transform(input_df)

    # * Mapping table is needed to join attribution-results and sid/order-id
    LOGGER.info('Creating mapping table from data_df and orderids_df')

    # * Successive step reads tables as feather
    data_df_dir = get_path(
        'data/interim',
        file_or_dir='dir',
        if_dir_not_exists='create',
    )
    LOGGER.info(f'Writing base_sessions_cj.feather to {data_df_dir}')
    path_to_bs = Path(data_df_dir) / 'base_sessions_cj.feather'
    data_df.reset_index(drop=True).to_feather(path_to_bs)

    LOGGER.info('Finished with data generation')
    return path_to_bs
