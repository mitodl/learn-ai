"""Tests for ai_chatbots utils"""

from uuid import UUID

from langchain_core.messages import HumanMessage
from named_enum import ExtendedEnum

from ai_chatbots.utils import add_message_ids, enum_zip, generate_message_id


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


def test_generate_message_id():
    """Test the generate_message_id function."""
    thread_id = "12345678-1234-5678-9abc-123456789abc"
    output_id = 1
    message = {"type": "HumanMessage", "content": "Hello world"}

    # Test basic functionality
    message_id = generate_message_id(thread_id, output_id, message)
    assert isinstance(message_id, str)
    assert UUID(message_id)  # Should be a valid UUID

    # Test deterministic behavior - same inputs should produce same output
    message_id2 = generate_message_id(thread_id, output_id, message)
    assert message_id == message_id2

    # Test different inputs produce different outputs
    different_thread = "87654321-4321-8765-cba9-987654321def"
    different_id = generate_message_id(different_thread, output_id, message)
    assert message_id != different_id

    different_output_id = generate_message_id(thread_id, 2, message)
    assert message_id != different_output_id

    different_message = {"type": "AIMessage", "content": "Hello world"}
    different_message_id = generate_message_id(thread_id, output_id, different_message)
    assert message_id != different_message_id


def test_add_message_ids_with_dicts():
    """Test add_message_ids function with dictionary messages."""
    thread_id = "12345678-1234-5678-9abc-123456789abc"
    output_id = 1

    # Test with messages that don't have IDs
    messages = [
        {"type": "HumanMessage", "content": "Hello"},
        {"type": "AIMessage", "content": "Hi there"},
    ]

    result = add_message_ids(thread_id, output_id, messages)

    # Should return the same list object
    assert result is messages

    # All messages should now have IDs
    for message in messages:
        assert "id" in message
        assert UUID(message["id"])  # Should be valid UUIDs

    # IDs should be deterministic
    messages2 = [
        {"type": "HumanMessage", "content": "Hello"},
        {"type": "AIMessage", "content": "Hi there"},
    ]
    add_message_ids(thread_id, output_id, messages2)

    for i, message in enumerate(messages):
        assert message["id"] == messages2[i]["id"]


def test_add_message_ids_preserves_existing_ids():
    """Test that add_message_ids preserves existing message IDs."""
    thread_id = "12345678-1234-5678-9abc-123456789abc"
    output_id = 1
    existing_id = "existing-id-123"

    messages = [
        {"type": "HumanMessage", "content": "Hello", "id": existing_id},
        {"type": "AIMessage", "content": "Hi there"},
    ]

    add_message_ids(thread_id, output_id, messages)

    # First message should keep its existing ID
    assert messages[0]["id"] == existing_id

    # Second message should get a new ID
    assert messages[1]["id"] != existing_id
    assert "id" in messages[1]


def test_add_message_ids_with_langchain_objects():
    """Test add_message_ids function with LangChain message objects."""
    thread_id = "12345678-1234-5678-9abc-123456789abc"
    output_id = 1

    # Test with LangChain message without ID
    message_without_id = HumanMessage(content="Hello")
    message_with_id = HumanMessage(content="Hi", id="existing-id")

    messages = [message_without_id, message_with_id]

    result = add_message_ids(thread_id, output_id, messages)

    # Should return the same list
    assert result is messages

    # First message should get an ID
    assert hasattr(message_without_id, "id")
    assert message_without_id.id is not None
    assert UUID(message_without_id.id)

    # Second message should keep its existing ID
    assert message_with_id.id == "existing-id"


def test_add_message_ids_empty_list():
    """Test add_message_ids with empty list."""
    thread_id = "12345678-1234-5678-9abc-123456789abc"
    output_id = 1

    messages = []
    result = add_message_ids(thread_id, output_id, messages)

    assert result is messages
    assert len(result) == 0
