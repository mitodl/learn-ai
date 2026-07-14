"""Tests for main.logging_filters."""

import logging

from main.logging_filters import DropFailedToDetachContext


def _record(msg):
    """Build a LogRecord with the given message."""
    return logging.LogRecord(
        name="opentelemetry.context",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_drops_failed_to_detach_context():
    """The detach-failure record is dropped."""
    assert not DropFailedToDetachContext().filter(_record("Failed to detach context"))


def test_keeps_other_records():
    """Any other record from the same logger passes through."""
    assert DropFailedToDetachContext().filter(_record("Failed to attach context"))
