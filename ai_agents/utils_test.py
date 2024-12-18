"""Tests for ai_agents utils"""

from named_enum import ExtendedEnum

from ai_agents.utils import enum_zip


def test_enum_zip():
    """Test the enum_zip function."""

    class TestEnum(ExtendedEnum):
        """enum test class"""

        foo = "bar"
        fizz = "buzz"
        hello = "world"

    new_enum = enum_zip("New_Enum", TestEnum)
    for item in new_enum:
        assert item.value == item.name
