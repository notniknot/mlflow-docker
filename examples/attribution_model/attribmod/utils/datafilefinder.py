import os
import sys
from pathlib import Path


def get_path(
    directory: str,
    filename: str = "",
    file_or_dir: str = 'file',
    if_dir_not_exists: str = None,
    raise_error: bool = True,
) -> str:
    """Return file path for given file in specific location.

    Args:
        directory (str): Subfolders containing the file
        name (str): Filename with extension
        file_or_dir (str): Check for 'file' or 'dir'
        if_dir_not_exists (str): Only for absolute paths! Pass 'create' if path should be created in case it does not exist.
        raise_error (bool): If true, application raises FileNotFoundError if not found

    Raises:
        FileNotFoundError: If file does not exist

    Returns:
        str: Path to specified file
    """

    if file_or_dir not in ['file', 'dir']:
        raise ValueError('No valid value for file_or_dir.')
    elif file_or_dir == 'file' and filename == '':
        raise ValueError('Filename not set.')
    elif file_or_dir == 'dir' and filename != '':
        raise ValueError('Filename needs to be empty.')

    paths = []
    if 'attribmod_state' not in os.environ or os.environ['attribmod_state'] == 'dev':
        paths += [
            Path(__file__).resolve() / directory / filename,
            Path(__file__).resolve().parent / directory / filename,
            Path(__file__).resolve().parent.parent / directory / filename,
            Path(__file__).resolve().parent.parent.parent / directory / filename,
        ]
    if 'attribmod_state' not in os.environ or os.environ['attribmod_state'] == 'prod':
        paths += [
            Path('/cosmos/attribmod') / directory / filename,
            Path(sys.prefix) / directory / filename,
            Path(sys.prefix) / 'attribmod' / directory / filename,
            Path(sys.exec_prefix) / directory / filename,
            Path(sys.exec_prefix) / 'attribmod' / directory / filename,
            Path(os.path.expanduser('~')) / 'attribmod' / directory / filename,
        ]

    for path in paths:
        if file_or_dir == 'file' and path.is_file():
            return str(path)
        elif file_or_dir == 'dir' and path.is_dir():
            return str(path)

    if file_or_dir == 'dir' and if_dir_not_exists == 'create':
        os.makedirs(directory, exist_ok=True)
        return directory

    if raise_error:
        raise FileNotFoundError('File not found in %s' % str(paths))
    else:
        return None
