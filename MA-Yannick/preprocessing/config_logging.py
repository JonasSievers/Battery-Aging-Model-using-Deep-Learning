import sys
from logging import *
import coloredlogs


logging_level = DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL


# Create a logger
basicConfig()
log = getLogger(__name__)
coloredlogs.install(logger=log)
log.propagate = False


coloredFormatter = coloredlogs.ColoredFormatter(
    fmt='%(message)s',
    level_styles=dict(
        debug=dict(color='white'),
        info=dict(color='blue'),
        warning=dict(color='yellow', bright=True),
        error=dict(color='red', bold=True, bright=True),
        critical=dict(color='black', bold=True, background='red'),
    ),
    field_styles=dict(
        name=dict(color='white'),
        asctime=dict(color='white'),
        funcName=dict(color='white'),
        lineno=dict(color='white'),
    )
)

ch = StreamHandler(stream=sys.stdout)
ch.setFormatter(fmt=coloredFormatter)
log.addHandler(hdlr=ch)
log.setLevel(level=logging_level)

# https://medium.com/geekculture/colored-logs-for-python-2973935a9b02
