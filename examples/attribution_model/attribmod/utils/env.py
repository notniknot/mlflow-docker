import logging
import os
from pathlib import Path

LOGGER = logging.getLogger()


def check_env():
    """
    Check current env-variables.
    """
    if 'attribmod_state' in os.environ and os.environ['attribmod_state'] == 'prod':
        LOGGER.setLevel(logging.INFO)
        base_path = Path('/cosmos/attribmod')
        paths = [
            base_path / 'data' / 'interim',
            base_path / 'data' / 'processed',
            base_path / 'data' / 'external',
        ]
        for path in paths:
            os.makedirs(path, exist_ok=True)
    else:
        os.environ['attribmod_state'] = 'dev'
        LOGGER.setLevel(logging.DEBUG)

    LOGGER.info(f'Detected {os.environ["attribmod_state"]}-environment')
