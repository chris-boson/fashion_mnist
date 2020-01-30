# @Author: Christfried Focke <christfriedf>
# @Date:   2020-01-29
# @Email:  christfried.focke@gmail.com


import logging
import sys


def get_logger(logging_level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(filename)s:%(lineno)d]'
    ))
    logger = logging.Logger('stdout_logger')
    logger.addHandler(handler)
    return logger