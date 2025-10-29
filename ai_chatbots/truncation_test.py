"""Tests for message truncation."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_chatbots.truncation import MessageTruncationNode


@pytest.fixture
def truncation_node():
    """Create a truncation node with max_human_messages=5."""
    return MessageTruncationNode(max_human_messages=5)


def test_truncation_with_few_messages(truncation_node):
    """Test that truncation doesn't affect conversations with few human messages."""
    messages = [
        SystemMessage(content="You are a helpful assistant", id=str(uuid4())),
        HumanMessage(content="Hello", id=str(uuid4())),
        AIMessage(content="Hi there!", id=str(uuid4())),
        HumanMessage(content="How are you?", id=str(uuid4())),
        AIMessage(content="I'm doing well!", id=str(uuid4())),
    ]

    result = truncation_node.invoke({"messages": messages})

    assert len(result["llm_input_messages"]) == 5
    assert result["llm_input_messages"] == messages


def test_truncation_with_many_human_messages(truncation_node):
    """Test that truncation keeps only last N human messages and their responses."""
    messages = [
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


def test_truncation_without_system_message(truncation_node):
    """Test truncation works when there's no system message."""
    messages = [
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
    ]

    result = truncation_node.invoke({"messages": messages})

    # Should have last 5 human messages + responses (10 messages total)
    assert len(result["llm_input_messages"]) == 10
    assert result["llm_input_messages"][0].content == "Question 2"
    assert result["llm_input_messages"][-1].content == "Answer 6"


def test_truncation_with_empty_messages(truncation_node):
    """Test truncation handles empty message list."""
    result = truncation_node.invoke({"messages": []})
    assert result["llm_input_messages"] == []


def test_truncation_with_exact_limit(truncation_node):
    """Test truncation when we have exactly max_human_messages."""
    messages = [
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
    ]

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


def test_truncation_preserves_message_order():
    """Test that message order is preserved after truncation."""
    truncation_node = MessageTruncationNode(max_human_messages=3)

    messages = [
        SystemMessage(content="System", id=str(uuid4())),
        HumanMessage(content="H1", id=str(uuid4())),
        AIMessage(content="A1", id=str(uuid4())),
        HumanMessage(content="H2", id=str(uuid4())),
        AIMessage(content="A2", id=str(uuid4())),
        HumanMessage(content="H3", id=str(uuid4())),
        AIMessage(content="A3", id=str(uuid4())),
        HumanMessage(content="H4", id=str(uuid4())),
        AIMessage(content="A4", id=str(uuid4())),
        HumanMessage(content="H5", id=str(uuid4())),
        AIMessage(content="A5", id=str(uuid4())),
    ]

    result = truncation_node.invoke({"messages": messages})

    # Should have system + last 3 human + their responses
    assert len(result["llm_input_messages"]) == 7
    contents = [msg.content for msg in result["llm_input_messages"]]
    assert contents == ["System", "H3", "A3", "H4", "A4", "H5", "A5"]


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


def test_find_nth_human_message_from_end():
    """Test the helper method for finding human message indices."""
    node = MessageTruncationNode(max_human_messages=3)

    messages = [
        AIMessage(content="AI intro", id=str(uuid4())),
        HumanMessage(content="H1", id=str(uuid4())),  # index 1
        AIMessage(content="A1", id=str(uuid4())),
        HumanMessage(content="H2", id=str(uuid4())),  # index 3
        AIMessage(content="A2", id=str(uuid4())),
        HumanMessage(content="H3", id=str(uuid4())),  # index 5
        AIMessage(content="A3", id=str(uuid4())),
    ]

    # Should find the 3rd-from-last human message (H1 at index 1)
    index = node.find_nth_human_message_from_end(messages, 3)
    assert index == 1
    assert messages[index].content == "H1"


def test_find_nth_human_message_not_enough():
    """Test helper when there aren't enough human messages."""
    node = MessageTruncationNode(max_human_messages=10)

    messages = [
        HumanMessage(content="H1", id=str(uuid4())),
        AIMessage(content="A1", id=str(uuid4())),
        HumanMessage(content="H2", id=str(uuid4())),
    ]

    # Should return 0 when there aren't enough human messages
    index = node.find_nth_human_message_from_end(messages, 10)
    assert index == 0
