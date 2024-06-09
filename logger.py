import logging


def get_logger(logger_name, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [P %(process)d] [T %(thread)d] %(name)s[line:%(lineno)d]: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger
