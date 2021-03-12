"""
Dieses Modul stellt Funktionen zur Verfügung, um Werte aus einer Konfigurationsdatei zu lesen.
"""

import os
from functools import lru_cache
from pathlib import Path
from shutil import copy2

import yaml
from attribmod.utils import my_logging
from attribmod.utils.datafilefinder import get_path

LOGGER = my_logging.logger(os.path.basename(__file__))


def get_config_file() -> str:
    """
    Die Methode versucht nach gewissen Mustern eine Konfigurationsdatei zu finden.

    Returns:
        Absoluter Pfad zur Konfigurationsdatei

    Raises:
        Falls keine Konfigurationsdatei gefunden wird, wird FileNotFoundError ausgelöst
    """
    name = 'configuration.yaml'
    config_file = None
    if 'WF_CONFIGURATION' in os.environ:
        config_file = os.environ['WF_CONFIGURATION']
    elif os.path.isfile(os.path.join(os.getcwd(), '.wf_config_yaml')):
        config_file = os.path.join(os.getcwd(), '.wf_config_yaml')
    elif os.path.isfile(os.path.join(os.path.dirname(__file__), name)):
        config_file = os.path.join(os.path.dirname(__file__), name)
    elif os.path.isfile(os.path.join(os.getcwd(), name)):
        config_file = os.path.join(os.getcwd(), name)
    elif os.path.isfile(os.path.join(os.getcwd(), '..', 'src', name)):
        config_file = os.path.join(os.getcwd(), '..', 'src', name)
    elif os.path.isfile(os.path.join(os.path.expanduser('~'), 'attribmod', 'src', name)):
        config_file = os.path.join(os.path.expanduser('~'), 'attribmod', 'src', name)
    elif os.path.isfile(os.path.join(os.path.dirname(__file__), '..', name)):
        config_file = os.path.join(os.path.dirname(__file__), '..', name)
    if config_file is None:
        raise FileNotFoundError('configuration file not found')
    config_file = os.path.abspath(config_file)
    LOGGER.debug('Configuration file is %s', config_file)
    return config_file


@lru_cache()
def get_config():
    """
    Liefert den Inhalt der Konfigurationsdatei.

    Returns:
        Inhalt der Konfiguration
    """
    config_file = get_path(directory='.', filename='configuration.yaml')
    LOGGER.debug('Configuration file is %s', config_file)

    with open(config_file, 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.SafeLoader)


def get_value(*args):
    """
    Liefert zu einer verschachtelten Angabe von Keys den zugehörigen Wert

    Args:
        *args: Keys

    Returns:
        Wert in der Konfigurationsdatei
    """
    conf = get_config()
    for arg in args:
        conf = conf[arg]
    return conf


def copy_config_to_local():
    """
    Erzeugt im aktuellen Verzeichnis eine Konfigurationsdatei.

    Returns:

    """
    src_file = Path(get_path('.', file_or_dir='dir')) / 'configuration_skeleton.yaml'
    trgt_file = Path(get_path('.', file_or_dir='dir')) / 'configuration.yaml'

    if os.path.exists(trgt_file):
        raise FileExistsError('Configuration file already exists (%s)' % trgt_file)
    copy2(src_file, trgt_file)


if __name__ == '__main__':
    copy_config_to_local()
