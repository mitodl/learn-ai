"""Tests for PostHog callback handler integration."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import Send

from ai_chatbots.posthog import TokenTrackingCallbackHandler, serialize_for_posthog


@pytest.fixture
def mock_bot(mocker):
    """Mock bot instance for testing"""
    bot = mocker.Mock()
    bot.user_id = "test_user"
    bot.thread_id = "test_thread"
    bot.JOB_ID = "TEST_JOB"
    return bot


@pytest.fixture
def mock_posthog_client(mocker):
    """Mock PostHog client"""
    return mocker.Mock()


def test_initialization(
    mock_posthog_client,
    mock_bot,
):
    """Test TokenTrackingCallbackHandler initialization with various parameters"""
    properties = {
        "$ai_model": "gpt-3.5-turbo",
        "$ai_provider": "openai",
    }
    handler = TokenTrackingCallbackHandler(
        model_name="openai/gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
        properties=properties,
    )

    for prop in properties:
        assert handler._properties[prop] == properties[prop]  # noqa: SLF001
    assert handler.input_tokens == 0
    assert handler.bot == mock_bot
    mock_posthog_client.capture.assert_called_once_with(
        event="$ai_trace",
        distinct_id=mock_bot.user_id,
        properties={
            "$ai_trace_id": mock_bot.thread_id,
            "$ai_span_name": mock_bot.JOB_ID,
            "botName": mock_bot.JOB_ID,
        },
    )


@pytest.mark.parametrize(
    ("messages", "expected_input_tokens"),
    [
        # Test with single message list containing human message
        ([[HumanMessage(content="Hello world")]], 3),
        # Test with nested message lists
        (
            [
                [
                    HumanMessage(content="Hello"),
                    SystemMessage(content="You are helpful"),
                ],
                [HumanMessage(content="How are you?")],
            ],
            6,
        ),
        # Test with AI message
        ([[AIMessage(content="I'm doing well, thank you!")]], 7),
        # Test empty messages
        ([[]], 0),
    ],
)
def test_on_chat_model_start_token_counting(
    mocker, mock_posthog_client, mock_bot, messages, expected_input_tokens
):
    """Test on_chat_model_start method with various message configurations"""
    # Mock litellm.token_counter to return predictable values
    mock_token_counter = mocker.patch("ai_chatbots.posthog.litellm.token_counter")
    mock_token_counter.return_value = expected_input_tokens
    # Mock the parent's on_chat_model_start to avoid calling real PostHog
    mocker.patch("posthog.ai.langchain.CallbackHandler.on_chat_model_start")

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
    )

    handler.on_chat_model_start(serialized={}, messages=messages, run_id="test_run")

    assert handler.input_tokens == expected_input_tokens
    if expected_input_tokens > 0:
        mock_token_counter.assert_called_once()


def test_on_chat_model_start_litellm_failure_fallback(
    mocker, mock_posthog_client, mock_bot
):
    """Test fallback to character-based estimation when litellm fails"""
    # Mock litellm.token_counter to raise an exception
    mock_token_counter = mocker.patch(
        "ai_chatbots.posthog.litellm.token_counter",
        side_effect=Exception("API error"),
    )
    mock_log = mocker.patch("ai_chatbots.posthog.log.exception")
    # Mock the parent's on_chat_model_start to avoid calling real PostHog
    mocker.patch("posthog.ai.langchain.CallbackHandler.on_chat_model_start")

    messages = [[HumanMessage(content="Hello world, this is a test message")]]

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
    )

    handler.on_chat_model_start(serialized={}, messages=messages, run_id="test_run")

    # Should fallback to character-based estimation: len("Hello world, this is a test message") // 4 = 8
    expected_tokens = len("Hello world, this is a test message") // 4
    assert handler.input_tokens == expected_tokens
    mock_log.assert_called_once()
    mock_token_counter.assert_called_once()


@pytest.mark.parametrize(
    ("response_text_data", "expected_output_tokens"),
    [
        # Test with single generation containing text
        ([["This is a response"]], 5),
        # Test with multiple generations
        ([["First part", "Second part"], ["Third part"]], 3),
        # Test with empty response
        ([[]], 0),
        # Test with generations without text attribute
        ([["no_text_attr"]], 0),
    ],
)
def test_on_llm_end_token_counting(
    mocker,
    mock_posthog_client,
    mock_bot,
    response_text_data,
    expected_output_tokens,
):
    """Test on_llm_end method with various response configurations"""
    # Create mock response generations based on text data
    response_generations = []
    for generation_group in response_text_data:
        generation_list = []
        for text_data in generation_group:
            if text_data == "no_text_attr":
                # Mock without text attribute
                generation_list.append(mocker.Mock(spec=[]))
            else:
                generation_list.append(mocker.Mock(text=text_data))
        response_generations.append(generation_list)

    # Create mock response object
    mock_response = mocker.Mock()
    mock_response.generations = response_generations

    # Mock litellm.token_counter for output
    mock_token_counter = mocker.patch("ai_chatbots.posthog.litellm.token_counter")
    mock_token_counter.return_value = expected_output_tokens
    # Mock the parent's on_llm_end to avoid calling real PostHog
    mocker.patch("posthog.ai.langchain.CallbackHandler.on_llm_end")

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
        properties={},
    )

    handler.on_llm_end(response=mock_response, run_id="test_run")

    assert handler._properties["$ai_output_tokens"] == expected_output_tokens  # noqa: SLF001
    if expected_output_tokens > 0:
        mock_token_counter.assert_called()


def test_on_llm_end_litellm_failure_fallback(mocker, mock_posthog_client, mock_bot):
    """Test fallback to character-based estimation when litellm fails for output"""
    # Create mock response with text
    mock_generation = mocker.Mock()
    mock_generation.text = "This is a test response with some content"
    mock_response = mocker.Mock()
    mock_response.generations = [[mock_generation]]

    # Mock litellm.token_counter to raise an exception
    mock_token_counter = mocker.patch(
        "ai_chatbots.posthog.litellm.token_counter",
        side_effect=Exception("API error"),
    )
    mock_log = mocker.patch("ai_chatbots.posthog.log.exception")
    # Mock the parent's on_llm_end to avoid calling real PostHog
    mocker.patch("posthog.ai.langchain.CallbackHandler.on_llm_end")

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
        properties={},
    )

    handler.on_llm_end(response=mock_response, run_id="test_run")

    # Should fallback to character-based estimation
    expected_tokens = len("This is a test response with some content") // 4
    assert handler._properties["$ai_output_tokens"] == expected_tokens  # noqa: SLF001
    mock_log.assert_called_once()
    mock_token_counter.assert_called_once()


def test_properties_update_on_llm_end(mocker, mock_posthog_client, mock_bot):
    """Test that properties are correctly updated in on_llm_end"""
    mock_generation = mocker.Mock()
    mock_generation.text = "Test response"
    mock_response = mocker.Mock()
    mock_response.generations = [[mock_generation]]

    mocker.patch("ai_chatbots.posthog.litellm.token_counter", return_value=10)
    # Mock the parent's on_llm_end to avoid calling real PostHog
    mocker.patch("posthog.ai.langchain.CallbackHandler.on_llm_end")

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
    )
    handler._properties = {"existing_key": "existing_value"}  # noqa: SLF001
    handler.input_tokens = 5

    handler.on_llm_end(response=mock_response, run_id="test_run")

    expected_properties = {
        "existing_key": "existing_value",
        "answer": "Test response",
        "$ai_input_tokens": 5,
        "$ai_output_tokens": 10,
        "$ai_trace_name": "TEST_JOB",
        "$ai_span_name": "TEST_JOB",
    }

    assert handler._properties == expected_properties  # noqa: SLF001


def test_multiple_human_messages_question_extraction(
    mocker, mock_posthog_client, mock_bot
):
    """Test that the last human message is used as the question"""
    mocker.patch("ai_chatbots.posthog.litellm.token_counter", return_value=10)
    # Mock the parent's on_chat_model_start to avoid calling real PostHog
    mock_parent = mocker.patch(
        "posthog.ai.langchain.CallbackHandler.on_chat_model_start"
    )

    # The current implementation incorrectly tries to iterate through nested lists
    # We need to provide the messages as a flat list for this test
    messages = [
        [HumanMessage(content="First question"), AIMessage(content="First answer")],
        [HumanMessage(content="Second question")],
    ]

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-3.5-turbo",
        client=mock_posthog_client,
        bot=mock_bot,
    )

    handler.on_chat_model_start(serialized={}, messages=messages, run_id="test_run")

    assert handler._properties["question"] == "Second question"  # noqa: SLF001
    mock_parent.assert_called_once()


@pytest.mark.parametrize(
    ("input_obj", "expected"),
    [
        # Primitive types
        ("test_string", "test_string"),
        (3.14, 3.14),
        (True, True),
        (None, None),
        # Lists and dicts
        ([1, "two", 3.0, True, None], [1, "two", 3.0, True, None]),
        ({"a": 1, "b": "two", "c": None}, {"a": 1, "b": "two", "c": None}),
        # LangChain Message objects
        (
            [
                HumanMessage(content="Hello", id="msg-1"),
                AIMessage(content="Hi", id="msg-2", tool_calls=[]),
            ],
            [
                {
                    "type": "human",
                    "role": "human",
                    "content": "Hello",
                    "id": "msg-1",
                    "additional_kwargs": {},
                },
                {
                    "type": "ai",
                    "role": "ai",
                    "content": "Hi",
                    "id": "msg-2",
                    "tool_calls": [],
                    "additional_kwargs": {},
                },
            ],
        ),
    ],
)
def test_serialize_for_posthog(input_obj, expected):
    """Test serialization of different objects/types"""
    result = serialize_for_posthog(input_obj)
    assert result == expected


def test_serialize_for_posthog_tool_calls():
    """Test serialization of AIMessage with tool calls"""
    msg = AIMessage(
        content="Using search tool",
        id="ai-msg-456",
        tool_calls=[
            {
                "name": "search_courses",
                "args": {"q": "data science", "resource_type": ["course"]},
                "id": "call_123",
                "type": "tool_call",
            }
        ],
    )
    result = serialize_for_posthog(msg)

    assert result == {
        "type": "ai",
        "role": "ai",
        "content": "Using search tool",
        "id": "ai-msg-456",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "tool_call",
                "function": {
                    "name": "search_courses",
                    "arguments": {"q": "data science", "resource_type": ["course"]},
                },
            },
        ],
        "additional_kwargs": {},
    }


def test_serialize_for_posthog_send_object():
    """Test serialization of LangGraph Send objects"""
    # Use actual Send object
    send_obj = [
        Send(node="tools", arg={"key": "value"}),
        Send(node="llm", arg={"messages": HumanMessage(content="Hello")}),
    ]

    result = serialize_for_posthog(send_obj)

    assert result == [
        {"node": "tools", "arg": {"key": "value"}},
        {
            "node": "llm",
            "arg": {
                "messages": {
                    "type": "human",
                    "role": "human",
                    "content": "Hello",
                    "id": None,
                    "additional_kwargs": {},
                }
            },
        },
    ]


def test_token_tracking_callback_handler_pop_run(mocker):
    """Test custom _pop_run_and_capture_trace_or_span method"""
    mock_bot = mocker.Mock()
    mock_bot.user_id = "test_user"
    mock_bot.thread_id = "test_thread"
    mock_bot.JOB_ID = "TEST_JOB"

    mock_posthog_client = mocker.Mock()

    mock_parent_method = mocker.patch(
        "posthog.ai.langchain.CallbackHandler._pop_run_and_capture_trace_or_span"
    )

    handler = TokenTrackingCallbackHandler(
        model_name="gpt-4", client=mock_posthog_client, bot=mock_bot
    )

    # Create list of Send objects
    send1 = Send(node="tool1", arg={"messages": HumanMessage(content="value1")})
    send2 = Send(node="tool2", arg={"messages": AIMessage(content="value2")})

    outputs = [send1, send2]

    handler._pop_run_and_capture_trace_or_span(uuid4(), uuid4(), outputs)  # noqa: SLF001

    serialized_output = mock_parent_method.call_args.args[2]
    assert len(serialized_output) == 2
    assert serialized_output == [
        {
            "node": "tool1",
            "arg": {
                "messages": {
                    "type": "human",
                    "role": "human",
                    "content": "value1",
                    "id": None,
                    "additional_kwargs": {},
                }
            },
        },
        {
            "node": "tool2",
            "arg": {
                "messages": {
                    "type": "ai",
                    "role": "ai",
                    "content": "value2",
                    "id": None,
                    "tool_calls": [],
                    "additional_kwargs": {},
                }
            },
        },
    ]
