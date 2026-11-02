# tests/test_logging.py
"""Logs go to the journal only (Q27): the manager logs to stderr, which
systemd sends to the journal, and journald timestamps every line."""
import logging
import sys

from manager.app import _setup_logging


def test_logging_goes_to_stderr_only(test_config):
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        _setup_logging(test_config)
        handlers = root.handlers[:]
    finally:
        root.handlers = saved
    assert len(handlers) == 1, handlers
    (handler,) = handlers
    assert type(handler) is logging.StreamHandler
    assert handler.stream is sys.stderr
    assert handler.formatter._fmt == "%(levelname)s %(name)s: %(message)s"
