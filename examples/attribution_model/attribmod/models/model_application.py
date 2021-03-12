import os
from pathlib import Path

from attribmod.data.abt_data import get_abt
from attribmod.models.attribution import model
from attribmod.utils import my_logging
from attribmod.utils.datafilefinder import get_path

LOGGER = my_logging.logger(os.path.basename(__file__))


def orchestrate_model_application(path_to_abt: str = None):
    """
    Orchestrate model application steps.
    """
    LOGGER.info('Starting orchestration of model application')

    LOGGER.info('Loading analytic_base_table.feather')
    if path_to_abt is None:
        path_to_abt = get_path('data/processed', 'analytic_base_table.feather')
    abt = get_abt(abt_file=path_to_abt)

    LOGGER.info('Applying model')
    result_df = model(abt, all_cj=True, use_fsc=1, use_macrodata=1, use_snbcorr=True)

    # * Successive step reads table as feather
    result_df_dir = get_path(
        'data/processed',
        file_or_dir='dir',
        if_dir_not_exists='create',
    )
    path_to_result = Path(result_df_dir) / 'attribution.feather'
    LOGGER.info(f'Writing attribution.feather to {path_to_result}')
    result_df.reset_index(drop=False).to_feather(path_to_result)
    LOGGER.info('Finished with model application')
    return path_to_result
