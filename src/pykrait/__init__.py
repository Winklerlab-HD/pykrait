import logging
import sys


class _StdoutHandler(logging.StreamHandler):
    """Writes to the current sys.stdout, which notebooks and tests replace after import."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


# pykrait reports its progress through the "pykrait" logger. By default the messages are
# printed to stdout as before; silence them with
# logging.getLogger("pykrait").setLevel(logging.WARNING), or remove this handler to
# route them through your own logging configuration.
_logger = logging.getLogger(__name__)
_handler = _StdoutHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_handler)
if _logger.level == logging.NOTSET:  # keep a level that was set before the import
    _logger.setLevel(logging.INFO)
_logger.propagate = False
