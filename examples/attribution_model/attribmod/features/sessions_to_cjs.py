import os
import pickle
from pathlib import Path

import pandas as pd
from attribmod.features.pipeline_steps.create_camp_fields import CreateCampFields
from attribmod.features.pipeline_steps.create_cumulative_count import CreateCumulativeCount
from attribmod.features.pipeline_steps.create_date_time_extracts import CreateDateTimeExtracts
from attribmod.features.pipeline_steps.create_journeys import CreateJourneys
from attribmod.features.pipeline_steps.encode_browser import EncodeBrowser
from attribmod.features.pipeline_steps.encode_first_page import EncodeFirstPage
from attribmod.features.pipeline_steps.encode_mobile_client import EncodeMobileClient
from attribmod.features.pipeline_steps.encode_os import EncodeOS
from attribmod.features.pipeline_steps.encode_product_page import EncodeProductPage
from attribmod.features.pipeline_steps.encode_touchpoints import EncodeTouchpoints
from attribmod.features.pipeline_steps.is_german_state import IsGermanState
from attribmod.features.pipeline_steps.partner_id_to_binary import PartnerIdToBinary
from attribmod.features.pipeline_steps.process_website_content import ProcessWebsiteContent
from attribmod.features.pipeline_steps.sort_df import SortDF
from attribmod.models.attrib_encoder import get_encoder, get_missing, store_encoder
from attribmod.utils import my_logging
from attribmod.utils.datafilefinder import get_path
from sklearn.pipeline import Pipeline

LOGGER = my_logging.logger(os.path.basename(__file__))


map_dict = {
    "kein_produkt": 0,
    "other": 0,
    "auswahl": 0.02,
    "eingabe": 0.1,
    "eingabe1": 0.1,
    "eingabe2": 0.1,
    "eingabe3": 0.125,
    "eingabe4": 0.15,
    "situation": 0.2,
    "ausgabe": 0.25,
    "partner": 0.4,
    "versicherungsumfang": 0.45,
    "vva": 0.5,
    "vorschaden": 0.55,
    "beruf": 0.55,
    "gesundheit": 0.6,
    "bank": 0.7,
    "ps": 0.85,
    "vitality": 0.9,
    "vitalityps": 0.95,
    "danke": 1,
}

map_kfz = {
    "1-versicherungsnehmer": 0.05,
    "2-fahrzeug-ergebnis": 0.1,
    "2-fahrzeug-ermitteln": 0.1,
    "3-zulassung": 0.15,
    "4-versicherungsbedarf": 0.2,
    "5-situation": 0.25,
    "6-schadenfreiheitsrabatt": 0.3,
    "7-fahrer": 0.35,
    "8-kilometerleistung": 0.4,
    "9-rabatte": 0.45,
    "10-tarifwahl": 0.5,
    "11-tarifdetails": 0.55,
    "12-berechnungsergebnis": 0.6,
    "13-vva": 0.65,
    "14-partner": 0.7,
    "15-kontakt": 0.75,
    "17-mcd-anlage": 0.85,
    "18-pruefen": 0.9,
    "19-dokumente": 0.95,
    "kfzeingabe": 0.1,
    "kfzeingabe1": 0.1,
    "kfzeingabe2": 0.2,
    "kfzeingabe3": 0.3,
    "kzfeingabe4": 0.4,
    "kfzausgabe": 0.5,
}

map_rlv = {
    "1-versicherungswunsch": 0.14,
    "2-versicherungsbedarf": 0.28,
    "3-versicherungsschutz": 0.42,
    "4-weitere-angaben": 0.58,
    "5-tarifwahl": 0.72,
    "6-tarifdetails": 0.86,
    "7-berechnungsergebnis": 1,
}

map_other = {
    "1-altersangabe": 0.1,
    "1-spezifikationen": 0.1,
    "2-tarifwahl": 0.2,
    "2-tarifwahl-sparangebot": 0.2,
    "2-versicherungsumfang": 0.2,
    "3-persondaten": 0.3,
    "3-grunddatenzahnstatus": 0.3,
    "4-erweiterte": 0.4,
    "4-berechnungsergebnis": 0.4,
    "6-beitragszahlung": 0.6,
    "7-pruefen": 0.7,
    "8-dokumente": 0.8,
    "kontakt": 0.75,
    "pruefen": 0.9,
}


keep_cols_raw = [
    'touchpoint',
    'wtr_visit_time_start',
    'wtr_region',
    'wtr_mobile_client',
    'wtr_os',
    'wtr_visit_time_month',
    'wtr_visit_time_dayofweek',
    'wtr_visit_time_hour',
    'camp',
    'camp_pca_0',
    'camp_pca_1',
    'camp_pca_2',
    'camp_pca_3',
    'first_page_0',
    'first_page_1',
    'first_page_2',
    'first_page_3',
    'first_page_4',
    'first_page_5',
    'wtr_browser_int',
    'web_partnerid',
    'wtr_avg_page_duration',
    'wtr_pages_in_session',
    'bouncer',
    'produktseite',
    'number_products',
    'content_depth',
]

makrodaten_raw = [
    'MACRODATA_DISPLAY_VISITS_0',
    'MACRODATA_SOCIAL_VISITS_0',
    'MACRODATA_PARTNER_VISITS_0',
    'MACRODATA_AFFILIATE_VISITS_0',
    'MACRODATA_ONLINEVIDEO_VISITS_0',
    'MACRODATA_SEANONBRAND_VISITS_0',
    'MACRODATA_DISPLAY_VISITS_1',
    'MACRODATA_SOCIAL_VISITS_1',
    'MACRODATA_PARTNER_VISITS_1',
    'MACRODATA_AFFILIATE_VISITS_1',
    'MACRODATA_ONLINEVIDEO_VISITS_1',
    'MACRODATA_SEANONBRAND_VISITS_1',
    'MACRODATA_DISPLAY_VISITS_2',
    'MACRODATA_SOCIAL_VISITS_2',
    'MACRODATA_PARTNER_VISITS_2',
    'MACRODATA_AFFILIATE_VISITS_2',
    'MACRODATA_ONLINEVIDEO_VISITS_2',
    'MACRODATA_SEANONBRAND_VISITS_2',
    'MACRODATA_DISPLAY_VISITS_3',
    'MACRODATA_SOCIAL_VISITS_3',
    'MACRODATA_PARTNER_VISITS_3',
    'MACRODATA_AFFILIATE_VISITS_3',
    'MACRODATA_ONLINEVIDEO_VISITS_3',
    'MACRODATA_SEANONBRAND_VISITS_3',
]


def _encoders2bytes(encoder: dict) -> dict:
    """Serializes all elements of a dict to a pickle byte string.

    Args:
        encoder (dict): Dict to serialize

    Returns:
        dict: Dict with serialized values
    """
    return {enc_type: pickle.dumps(enc) for enc_type, enc in encoder.items()}


def _collect_params_to_store(abt_pipeline: Pipeline, missing: str):
    encoder = {
        'touchpoint': abt_pipeline.named_steps['touchpoint'].encoder,
        'mobile_client': abt_pipeline.named_steps['mobile_client'].encoder,
        'os': abt_pipeline.named_steps['os'].encoder,
        'first_page': abt_pipeline.named_steps['first_page'].encoder,
        'produktseite': abt_pipeline.named_steps['produktseite'].encoder,
        'kampagne_mlb': abt_pipeline.named_steps['kampagne'].mlb,
        'kampagne_pca': abt_pipeline.named_steps['kampagne'].pca,
        'kampagne_pca_cols': abt_pipeline.named_steps['kampagne'].pca_cols,
    }
    store_encoder([missing, encoder])


def orchestrate_data_processing(train: bool = False, path_to_bs: str = None):
    """Orchestrate data processing steps.

    Args:
        train (bool, optional): If True, refits the encoders. Defaults to False.
    """
    LOGGER.info('Starting orchestration of data processing')

    path = (
        get_path('data/interim', 'base_sessions_cj.feather') if path_to_bs is None else path_to_bs
    )
    LOGGER.info(f'Loading base_sessions_cj.feather from {path}')

    base_sessions_df = pd.read_feather(path)

    # * Column names need to be lower-case
    makrodaten = [mdname.lower() for mdname in makrodaten_raw]

    keep_cols = []
    keep_cols += keep_cols_raw
    keep_cols += makrodaten

    map_dict.update(map_kfz)
    map_dict.update(map_rlv)
    map_dict.update(map_other)

    abt_pipeline = Pipeline(
        steps=[
            ('sort', SortDF()),
            ('no', CreateCumulativeCount()),
            ('touchpoint', EncodeTouchpoints()),
            ('region', IsGermanState()),
            ('mobile_client', EncodeMobileClient()),
            ('os', EncodeOS()),
            ('visit_time', CreateDateTimeExtracts()),
            ('first_page', EncodeFirstPage()),
            ('browser', EncodeBrowser()),
            ('partnerid', PartnerIdToBinary()),
            ('produktseite', EncodeProductPage()),
            ('web_seiten_content', ProcessWebsiteContent()),
            ('kampagne', CreateCampFields()),
            ('create_journeys', CreateJourneys()),
        ]
    )

    # * Encoders can only be passed into the pipeline via pickle streams
    encoder = _encoders2bytes(get_encoder())
    missing = get_missing()
    encoder_params = {
        'touchpoint__encoder': encoder['touchpoint'],
        'touchpoint__missing': missing,
        'mobile_client__encoder': encoder['mobile_client'],
        'mobile_client__missing': missing,
        'os__encoder': encoder['os'],
        'os__missing': missing,
        'first_page__encoder': encoder['first_page'],
        'first_page__missing': missing,
        'produktseite__encoder': encoder['produktseite'],
        'produktseite__missing': missing,
        'kampagne__mlb': encoder['kampagne_mlb'],
        'kampagne__pca': encoder['kampagne_pca'],
        'kampagne__pca_cols': encoder['kampagne_pca_cols'],
    }
    if train is True:
        encoder_params = {
            k: v if k.endswith('__missing') else None for k, v in encoder_params.items()
        }
    other_params = {
        'web_seiten_content__map_dict': map_dict,
        'create_journeys__keep_cols': keep_cols,
    }
    abt_pipeline.set_params(**encoder_params, **other_params)

    LOGGER.info('Executing pipeline')
    abt_df = abt_pipeline.fit_transform(base_sessions_df)

    if train is True:
        LOGGER.info('Storing encoders')
        _collect_params_to_store(abt_pipeline, missing)

    # * Successive step reads table as feather
    abt_df_dir = get_path(
        'data/processed',
        file_or_dir='dir',
        if_dir_not_exists='create',
    )
    path_to_abt = Path(abt_df_dir) / 'analytic_base_table.feather'
    LOGGER.info(f'Writing analytic_base_table.feather to {abt_df_dir}')
    abt_df.reset_index(drop=True).to_feather(path_to_abt)
    LOGGER.info('Finished with data processing')
    return path_to_abt
