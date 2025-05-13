import logging
import sys


def setup_logger(logger_name=None, level=logging.INFO, stream=sys.stdout):
    """
    Sets up the logger configuration.

    :param logger_name: Name of the logger to set up. If None, sets up the root logger.
    :param level: Logging level.
    :param stream: Stream to log to, default is sys.stdout.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    logger.handlers = []

    # Add the new handler
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    return logger
