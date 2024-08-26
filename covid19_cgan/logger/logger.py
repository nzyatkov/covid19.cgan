import logging
import time
from pathlib import Path

_logger = None
_log_format = f"%(asctime)s - [%(levelname)s] - %(message)s | %(filename)s - %(funcName)s(%(lineno)d)"
_log_level = logging.INFO


def get_file_handler(name):
    global _log_level

    log_dir = Path('data', 'logs')
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", mode='w')
    file_handler.setLevel(_log_level)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler

def get_stream_handler():
    global _log_level

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(_log_level)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler

def get_logger():
    global _logger
    global _log_level

    name = 'cgan_logger'

    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(_log_level)
        _logger.addHandler(get_file_handler(name))
        _logger.addHandler(get_stream_handler())
        logging.Formatter.converter = time.gmtime

    return _logger