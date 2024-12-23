"""Utility functions for ai chat agents"""

import logging
from enum import Enum

from named_enum import ExtendedEnum

log = logging.getLogger(__name__)


def enum_zip(label: str, enum: ExtendedEnum) -> type[Enum]:
    """
    Create a new Enum with both name and value equal to
    the names of the given ExtendedEnum.

    Args:
        label: The label for the new Enum
        enum: The Enum to use as a basis for the new Enum

    Returns:
        A new Enum with the names of the given Enum as both name and value

    """
    return Enum(label, dict(zip(enum.names(), enum.names())))
