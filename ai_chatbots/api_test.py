"""Tests for ai_chatbots/api.py"""

import json
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from ai_chatbots import factories
from ai_chatbots.api import (
    MessageTruncationNode,
    _should_create_checkpoint,
    create_tutor_checkpoints,
    create_tutorbot_output_and_checkpoints,
)
from ai_chatbots.chatbots import SystemMessage
from ai_chatbots.models import DjangoCheckpoint, TutorBotOutput, UserChatSession


@pytest.fixture
def truncation_node():
    """Create a truncation node with max_human_messages=5."""
    return MessageTruncationNode(max_human_messages=5)


@pytest.fixture
def sample_messages():
    """Create a list of messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant", id=str(uuid4())),
        HumanMessage(content="Question 1", id=str(uuid4())),
        AIMessage(content="Answer 1", id=str(uuid4())),
        HumanMessage(content="Question 2", id=str(uuid4())),
        AIMessage(content="Answer 2", id=str(uuid4())),
        HumanMessage(content="Question 3", id=str(uuid4())),
        AIMessage(content="Answer 3", id=str(uuid4())),
        HumanMessage(content="Question 4", id=str(uuid4())),
        AIMessage(content="Answer 4", id=str(uuid4())),
        HumanMessage(content="Question 5", id=str(uuid4())),
        AIMessage(content="Answer 5", id=str(uuid4())),
        HumanMessage(content="Question 6", id=str(uuid4())),
        AIMessage(content="Answer 6", id=str(uuid4())),
        HumanMessage(content="Question 7", id=str(uuid4())),
        AIMessage(content="Answer 7", id=str(uuid4())),
    ]


@pytest.mark.django_db
def test_create_tutor_checkpoints_no_session():
    """Test create_tutor_checkpoints when UserChatSession doesn't exist."""

    # Use a thread_id that definitely doesn't exist
    thread_id = "99999999-9999-9999-9999-999999999999"

    # Ensure no session exists for this thread_id
    assert not UserChatSession.objects.filter(thread_id=thread_id).exists()

    chat_json = '{"chat_history": [{"type": "HumanMessage", "content": "Hello"}]}'

    result = create_tutor_checkpoints(thread_id, chat_json)

    assert result == []


@pytest.mark.django_db
def test_create_tutor_checkpoints_empty_messages():
    """Test create_tutor_checkpoints with empty message history."""

    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id)

    chat_json = '{"chat_history": []}'

    result = create_tutor_checkpoints(thread_id, chat_json)

    assert result == []


@pytest.mark.django_db
def test_create_tutor_checkpoints_with_tool_messages():
    """Test create_tutor_checkpoints filters out ToolMessage types."""
    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id)

    chat_json = """
    {
        "chat_history": [
            {"type": "ToolMessage", "content": "Tool result", "id": "msg0"},
            {"type": "HumanMessage", "content": "Testing 123", "id": "msg1"}
        ]
    }
    """

    result = create_tutor_checkpoints(thread_id, chat_json)

    assert len(result) == 1
    checkpoint_meta = json.dumps(result[0].metadata)
    assert "Tool result" not in checkpoint_meta
    assert "Testing 123" in checkpoint_meta


@pytest.mark.django_db
def test_create_tutor_checkpoints_with_valid_messages():
    """Test create_tutor_checkpoints creates checkpoints for valid messages."""
    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id)

    chat_json = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},
            {"type": "AIMessage", "content": "Hi there", "id": "msg2"},
        ]
    }

    result = create_tutor_checkpoints(thread_id, chat_json)

    assert len(result) == 2
    assert all(isinstance(checkpoint, DjangoCheckpoint) for checkpoint in result)

    # Check that checkpoints were actually created in DB
    saved_checkpoints = DjangoCheckpoint.objects.filter(thread_id=thread_id)
    assert saved_checkpoints.count() == 2


@pytest.mark.django_db
def test_create_tutor_checkpoints_new_messages_only():
    """Test create_tutor_checkpoints only creates checkpoints for new messages."""
    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id, user=None)

    # Previous chat with one message
    previous_chat = json.dumps(
        {"chat_history": [{"type": "HumanMessage", "content": "Hello", "id": "msg1"}]}
    )

    # New chat with the same message plus a new one
    new_chat = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},  # Existing
            {"type": "AIMessage", "content": "Hi there", "id": "msg2"},  # New
        ]
    }

    result = create_tutor_checkpoints(
        thread_id, new_chat, previous_chat_json=previous_chat
    )

    # Should only create checkpoint for the new message
    assert len(result) == 1


@pytest.mark.django_db
async def test_create_tutorbot_output_and_checkpoints():
    """Test create_tutorbot_output_and_checkpoints creates both objects."""
    thread_id = str(uuid4())
    await sync_to_async(factories.UserChatSessionFactory.create)(thread_id=thread_id)

    chat_json = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},
            {"type": "AIMessage", "content": "Hi there", "id": "msg2"},
        ]
    }
    edx_module_id = "test-module"

    output, checkpoints = await create_tutorbot_output_and_checkpoints(
        thread_id, chat_json, edx_module_id
    )

    # Check TutorBotOutput was created
    assert isinstance(output, TutorBotOutput)
    assert output.thread_id == thread_id
    assert output.edx_module_id == edx_module_id

    # Check checkpoints were created
    assert len(checkpoints) == 2
    assert all(isinstance(cp, DjangoCheckpoint) for cp in checkpoints)

    # Verify they're saved in DB
    assert (
        await sync_to_async(TutorBotOutput.objects.filter(thread_id=thread_id).count)()
        == 1
    )
    assert (
        await sync_to_async(
            DjangoCheckpoint.objects.filter(thread_id=thread_id).count
        )()
        == 2
    )


@pytest.mark.django_db
async def test_create_tutorbot_output_and_checkpoints_with_previous():
    """Test create_tutorbot_output_and_checkpoints compares with previous output."""
    thread_id = str(uuid4())
    await sync_to_async(factories.UserChatSessionFactory.create)(
        thread_id=thread_id, user=None
    )

    # Create a previous output
    previous_chat = json.dumps(
        {"chat_history": [{"type": "HumanMessage", "content": "Hello", "id": "msg1"}]}
    )
    await sync_to_async(TutorBotOutput.objects.create)(
        thread_id=thread_id, chat_json=previous_chat, edx_module_id="test"
    )

    # New chat with additional message
    new_chat = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},  # Existing
            {"type": "AIMessage", "content": "Hi", "id": "msg2"},  # New
        ]
    }

    output, checkpoints = await create_tutorbot_output_and_checkpoints(
        thread_id, new_chat, "test-module"
    )

    # Should create new output and only checkpoint for new message
    assert len(checkpoints) == 1
    assert (
        await sync_to_async(TutorBotOutput.objects.filter(thread_id=thread_id).count)()
        == 2
    )
    assert (
        await sync_to_async(TutorBotOutput.objects.filter(id=output.id).exists)()
        is True
    )


def test_should_create_checkpoint():
    """Test the _should_create_checkpoint function filters correctly."""
    # Should create checkpoint for HumanMessage
    human_msg = {"type": "HumanMessage", "content": "Hello"}
    assert _should_create_checkpoint(human_msg) is True

    # Should create checkpoint for AIMessage without tool_calls
    ai_msg = {"type": "AIMessage", "content": "Hi there"}
    assert _should_create_checkpoint(ai_msg) is True

    # Should create checkpoint for AIMessage with empty tool_calls
    ai_msg_empty_tools = {"type": "AIMessage", "content": "Hi", "tool_calls": []}
    assert _should_create_checkpoint(ai_msg_empty_tools) is True

    # Should create checkpoint for AIMessage with null tool_calls
    ai_msg_null_tools = {"type": "AIMessage", "content": "Hi", "tool_calls": None}
    assert _should_create_checkpoint(ai_msg_null_tools) is True

    # Should NOT create checkpoint for ToolMessage
    tool_msg = {"type": "ToolMessage", "content": "Tool result"}
    assert _should_create_checkpoint(tool_msg) is False

    # Should NOT create checkpoint for AIMessage with tool_calls
    ai_msg_with_tools = {
        "type": "AIMessage",
        "content": "Let me help you",
        "tool_calls": [{"name": "search", "args": {"query": "test"}}],
    }
    assert _should_create_checkpoint(ai_msg_with_tools) is False


@pytest.mark.django_db
def test_create_tutor_checkpoints_filters_ai_messages_with_tool_calls():
    """Test that create_tutor_checkpoints filters out AI messages with tool_calls."""
    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id)

    chat_json = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},
            {
                "type": "AIMessage",
                "content": "Let me search",
                "id": "msg2",
                "tool_calls": [{"name": "search"}],
            },
            {"type": "ToolMessage", "content": "Search results", "id": "msg3"},
            {"type": "AIMessage", "content": "Here are the results", "id": "msg4"},
        ]
    }

    result = create_tutor_checkpoints(thread_id, chat_json)

    # Should only create checkpoints for HumanMessage and final AIMessage (without tool_calls)
    assert len(result) == 2

    # Verify in database
    saved_checkpoints = DjangoCheckpoint.objects.filter(thread_id=thread_id)
    assert saved_checkpoints.count() == 2


@pytest.mark.django_db
def test_create_tutor_checkpoints_includes_metadata():
    """Test that create_tutor_checkpoints includes chat metadata in checkpoint writes."""
    thread_id = str(uuid4())
    factories.UserChatSessionFactory.create(thread_id=thread_id)

    # Include metadata in chat_json
    chat_json = {
        "chat_history": [
            {"type": "HumanMessage", "content": "Hello", "id": "msg1"},
            {"type": "AIMessage", "content": "Hi there", "id": "msg2"},
        ],
        "metadata": {
            "user_id": "test_user_123",
            "course_id": "course_456",
            "custom_field": "custom_value",
        },
    }

    result = create_tutor_checkpoints(thread_id, chat_json)

    assert len(result) == 2

    # Check that metadata is included in checkpoint writes
    for idx, checkpoint in enumerate(result):
        metadata = checkpoint.metadata
        writes = metadata.get("writes", {})

        # The writes should include the tutor metadata
        key = next(iter(writes.keys()))
        assert key == ("__start__" if idx == 0 else "agent")
        start_writes = writes.get(key)
        assert start_writes["user_id"] == "test_user_123"
        assert start_writes["course_id"] == "course_456"
        assert start_writes["custom_field"] == "custom_value"

        # Messages should still be present
        assert len(start_writes["messages"]) == 1
        assert (
            start_writes["messages"][0]
            == checkpoint.checkpoint["channel_values"]["messages"][idx]
        )


def test_create_langchain_message_id_handling():
    """Test that _create_langchain_message preserves existing IDs or generates new ones."""
    from ai_chatbots.api import _create_langchain_message

    message = {"type": "HumanMessage", "content": "Test message", "id": str(uuid4())}

    result = _create_langchain_message(message)

    assert result["kwargs"]["id"] == message["id"]
    assert result["kwargs"]["type"] == "human"
    assert result["kwargs"]["content"] == "Test message"


@pytest.mark.django_db
@pytest.mark.parametrize("has_previous_checkpoint", [True, False])
def test_create_tutor_checkpoints_step_calculation(has_previous_checkpoint):
    """Test that step calculation works correctly with or without previous checkpoints."""
    thread_id = str(uuid4())

    factories.UserChatSessionFactory.create(thread_id=thread_id)

    if has_previous_checkpoint:
        initial_chat_data = {
            "chat_history": [
                {
                    "type": "HumanMessage",
                    "content": "Initial message",
                    "id": str(uuid4()),
                },
                {"type": "AIMessage", "content": "Response", "id": str(uuid4())},
            ],
            "user_id": "test_user",
            "course_id": "test_course",
        }

        previous_checkpoints = create_tutor_checkpoints(thread_id, initial_chat_data)
        assert len(previous_checkpoints) == 2
        assert previous_checkpoints[0].metadata["step"] == 0
    else:
        initial_chat_data = None

    chat_data = {
        "chat_history": [
            {"type": "HumanMessage", "content": "New message", "id": str(uuid4())},
            {"type": "AIMessage", "content": "Response", "id": str(uuid4())},
        ],
        "user_id": "test_user",
        "course_id": "test_course",
    }

    result = create_tutor_checkpoints(
        thread_id, chat_data, previous_chat_json=initial_chat_data
    )

    assert len(result) == 2
    assert result[0].metadata["step"] == (2 if has_previous_checkpoint else 0)
    assert result[1].metadata["step"] == (3 if has_previous_checkpoint else 1)


def test_truncation_with_few_messages(truncation_node, sample_messages):
    """Test that truncation doesn't affect conversations with few human messages."""
    messages = sample_messages[:5]  # System + 2 human + 2 AI

    result = truncation_node.invoke({"messages": messages})

    assert len(result["llm_input_messages"]) == 5
    assert result["llm_input_messages"] == messages


def test_truncation_with_many_human_messages(truncation_node, sample_messages):
    """Test that truncation keeps only last N human messages and their responses."""
    messages = sample_messages  # All 15 messages

    result = truncation_node.invoke({"messages": messages})

    # Should have system message + last 5 human messages + their responses (10 messages)
    assert len(result["llm_input_messages"]) == 11  # system + 5 human + 5 AI
    assert isinstance(result["llm_input_messages"][0], SystemMessage)

    # Should have kept questions 3-7
    assert result["llm_input_messages"][1].content == "Question 3"
    assert result["llm_input_messages"][2].content == "Answer 3"
    assert result["llm_input_messages"][-2].content == "Question 7"
    assert result["llm_input_messages"][-1].content == "Answer 7"


def test_truncation_with_tool_calls():
    """Test that truncation includes tool calls when human message triggered them."""
    truncation_node = MessageTruncationNode(max_human_messages=2)

    tool_call_id = str(uuid4())
    messages = [
        SystemMessage(content="You are a helpful assistant", id=str(uuid4())),
        HumanMessage(content="Question 1", id=str(uuid4())),
        AIMessage(content="Answer 1", id=str(uuid4())),
        HumanMessage(content="Question 2 needs search", id=str(uuid4())),
        AIMessage(
            content="Let me search",
            id=str(uuid4()),
            tool_calls=[{"id": tool_call_id, "name": "search", "args": {}}],
        ),
        ToolMessage(
            content="Search results", id=str(uuid4()), tool_call_id=tool_call_id
        ),
        AIMessage(content="Based on search: Answer 2", id=str(uuid4())),
        HumanMessage(content="Question 3", id=str(uuid4())),
        AIMessage(content="Answer 3", id=str(uuid4())),
    ]

    result = truncation_node.invoke({"messages": messages})

    # Should have system + last 2 human messages + all their responses including tools
    assert len(result["llm_input_messages"]) >= 7  # system + Q2 + 3 AI msgs + Q3 + A3
    assert isinstance(result["llm_input_messages"][0], SystemMessage)
    assert result["llm_input_messages"][1].content == "Question 2 needs search"
    assert result["llm_input_messages"][-2].content == "Question 3"
    assert result["llm_input_messages"][-1].content == "Answer 3"


def test_truncation_without_system_message(truncation_node, sample_messages):
    """Test truncation works when there's no system message."""
    messages = sample_messages[1:13]  # Skip system message, use Q1-Q6 + A1-A6

    result = truncation_node.invoke({"messages": messages})

    # Should have last 5 human messages + responses (10 messages total)
    assert len(result["llm_input_messages"]) == 10
    assert result["llm_input_messages"][0].content == "Question 2"
    assert result["llm_input_messages"][-1].content == "Answer 6"


def test_truncation_with_empty_messages(truncation_node):
    """Test truncation handles empty message list."""
    result = truncation_node.invoke({"messages": []})
    assert result["llm_input_messages"] == []


def test_truncation_with_exact_limit(truncation_node, sample_messages):
    """Test truncation when we have exactly max_human_messages."""
    messages = sample_messages[:11]  # System + Q1-Q5 + A1-A5

    result = truncation_node.invoke({"messages": messages})

    # Should keep all messages (no truncation needed)
    assert len(result["llm_input_messages"]) == 11
    assert result["llm_input_messages"] == messages


def test_truncation_with_only_ai_messages():
    """Test truncation when there are no human messages."""
    truncation_node = MessageTruncationNode(max_human_messages=5)

    messages = [
        SystemMessage(content="You are a helpful assistant", id=str(uuid4())),
        AIMessage(content="Welcome!", id=str(uuid4())),
        AIMessage(content="How can I help?", id=str(uuid4())),
    ]

    result = truncation_node.invoke({"messages": messages})

    # Should keep all messages since there are no human messages to count
    assert len(result["llm_input_messages"]) == 3
    assert result["llm_input_messages"] == messages


def test_truncation_with_multiple_ai_responses():
    """Test truncation when AI sends multiple messages per human question."""
    truncation_node = MessageTruncationNode(max_human_messages=2)

    messages = [
        SystemMessage(content="System", id=str(uuid4())),
        HumanMessage(content="Question 1", id=str(uuid4())),
        AIMessage(content="Let me think...", id=str(uuid4())),
        AIMessage(content="Here's part 1", id=str(uuid4())),
        AIMessage(content="And part 2", id=str(uuid4())),
        HumanMessage(content="Question 2", id=str(uuid4())),
        AIMessage(content="Answer 2", id=str(uuid4())),
    ]

    result = truncation_node.invoke({"messages": messages})

    # Should include Question 1 and all its AI responses, plus Question 2
    # System + Q1 + 3 AI + Q2 + A2 = 7 messages
    assert len(result["llm_input_messages"]) == 7
    assert result["llm_input_messages"][0].content == "System"
    assert result["llm_input_messages"][1].content == "Question 1"
    assert result["llm_input_messages"][-2].content == "Question 2"
    assert result["llm_input_messages"][-1].content == "Answer 2"


def test_find_nth_human_message_from_end(sample_messages):
    """Test the helper method for finding human message indices."""
    node = MessageTruncationNode(max_human_messages=3)
    messages = [AIMessage(content="Hello", id=str(uuid4()))] + sample_messages[
        1:7
    ]  # Question 1-3, Answer 1-3

    # Should find the 3rd-from-last human message (Question 1 at index 0)
    index = node.find_nth_human_message_from_end(messages, 3)
    assert index == 1
    assert messages[index].content == "Question 1"


def test_find_nth_human_message_not_enough(sample_messages):
    """Test helper when there aren't enough human messages."""
    node = MessageTruncationNode(max_human_messages=10)
    messages = sample_messages[1:5]  # Question 1, Answer 1, Question 2, Answer 2

    # Should return 0 when there aren't enough human messages
    index = node.find_nth_human_message_from_end(messages, 10)
    assert index == 0
