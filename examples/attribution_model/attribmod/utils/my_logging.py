import logging

DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

_logger = None
_console_handler = None


def _init_logger():
    global _logger, _console_handler
    if _logger is None:
        _logger = logging.getLogger()
        _logger.handlers = []
        # _logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        _console_handler = logging.StreamHandler()
        _console_handler.setLevel(logging.DEBUG)
        # ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        _console_handler.setFormatter(formatter)

        # add ch to logger
        _logger.addHandler(_console_handler)


def logger(obj):
    global _logger
    if _logger is None:
        _init_logger()

    if isinstance(obj, str):
        mylogger = logging.getLogger(str(obj))
    elif obj.__module__ == '__main__':
        mylogger = logging.getLogger(obj.__class__.__name__)
    else:
        mylogger = logging.getLogger(obj.__module__ + "." + obj.__class__.__name__)

    return mylogger
