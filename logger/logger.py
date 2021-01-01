import logging
import os
import coloredlogs


def get_logger(tag: str) -> logging.Logger:
    log = logging.getLogger(tag)
    log.setLevel(os.environ.get("LOG_LEVEL", "WARNING"))
    coloredlogs.install(level=log.level, logger=log)
    log.propagate = False

    return log
