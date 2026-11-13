"""Tests for evaluation.base module."""

from unittest.mock import Mock

import pytest
from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_chatbots.evaluation.base import (
    SAMPLE_SEED,
    BaseBotEvaluator,
    EvaluationConfig,
    RunFailures,
    TestCaseSpec,
    sample_cases,
)


class TestBaseBotEvaluator:
    """Test cases for BaseBotEvaluator abstract base class."""

    def test_test_case_spec_creation(self):
        """Test TestCaseSpec creation and serialization."""
        spec = TestCaseSpec(
            question="What is Python?",
            expected_output="Python is a programming language",
            expected_tools=["search"],
            metadata={"difficulty": "easy"},
        )

        assert spec.question == "What is Python?"
        assert spec.expected_output == "Python is a programming language"
        assert spec.expected_tools == ["search"]
        assert spec.metadata == {"difficulty": "easy"}

        # Test to_dict method
        spec_dict = spec.to_dict()
        expected_dict = {
            "question": "What is Python?",
            "expected_output": "Python is a programming language",
            "expected_tools": ["search"],
            "metadata": {"difficulty": "easy"},
        }
        assert spec_dict == expected_dict

    def test_test_case_spec_defaults(self):
        """Test TestCaseSpec with default values."""
        spec = TestCaseSpec(question="Test question")

        assert spec.question == "Test question"
        assert spec.expected_output is None
        assert spec.expected_tools is None
        assert spec.metadata is None

        # Test to_dict with defaults
        spec_dict = spec.to_dict()
        expected_dict = {
            "question": "Test question",
            "expected_output": None,
            "expected_tools": [],
            "metadata": {},
        }
        assert spec_dict == expected_dict

    def test_evaluation_config_creation(self, mocker):
        """Test EvaluationConfig creation with all parameters."""
        mock_metrics = [mocker.Mock(), mocker.Mock()]
        metric_thresholds = {"AnswerRelevancy": 0.7, "Hallucination": 0.0}
        config = EvaluationConfig(
            models=["gpt-4", "gpt-3.5"],
            evaluation_model="gpt-4",
            metrics=mock_metrics,
            metric_thresholds=metric_thresholds,
            confident_api_key="test-key",
        )

        assert config.models == ["gpt-4", "gpt-3.5"]
        assert config.evaluation_model == "gpt-4"
        assert config.metrics == mock_metrics
        assert config.metric_thresholds == metric_thresholds
        assert config.confident_api_key == "test-key"

    def test_evaluation_config_optional_api_key(self):
        """Test EvaluationConfig without API key."""
        config = EvaluationConfig(
            models=["gpt-4"],
            evaluation_model="gpt-4",
            metrics=[],
            metric_thresholds={},
        )

        assert config.confident_api_key is None


class ConcreteBotEvaluator(BaseBotEvaluator):
    """Concrete implementation for testing abstract methods."""

    def __init__(
        self, bot_class, bot_name: str, *, data_file: str | None = None, mocker=None
    ):
        """Initialize with optional mocker for testing."""
        super().__init__(bot_class, bot_name, data_file=data_file)
        self._mocker = mocker

    def load_test_cases(self):
        return [
            TestCaseSpec(
                question="Test question",
                expected_output="Test output",
                expected_tools=["test_tool"],
                metadata={"extra_state": {"test": "value"}},
            )
        ]

    def validate_test_case(self, test_case):
        return test_case.question is not None

    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
    ):
        """Create mock bot instance."""
        _ = model, test_case, instructions  # Unused parameters for testing
        mock_bot = self._mocker.Mock() if self._mocker else Mock()
        mock_bot.config = {"configurable": {"test": "config"}}
        return mock_bot

    async def collect_response(self, chatbot, test_case):
        """Collect mock response."""
        _ = chatbot, test_case  # Unused parameters for testing

        # Create mock tool call with proper string name
        tool_call_mock = self._mocker.Mock() if self._mocker else Mock()
        tool_call_mock.function.name = "test_tool"
        tool_call_mock.function.arguments = '{"arg": "value"}'

        return {
            "messages": [
                HumanMessage(content="User message"),
                AIMessage(
                    content="Tool call",
                    additional_kwargs={"tool_calls": [tool_call_mock]},
                ),
                AIMessage(content='{"results": [{"chunk_content": "Test content"}]}'),
                AIMessage(content="Bot response"),
            ]
        }


class TestConcreteBotEvaluator:
    """Test cases for concrete BaseBotEvaluator implementation."""

    @pytest.fixture
    def evaluator(self, mocker):
        """Create a concrete evaluator for testing."""
        mock_bot_class = mocker.Mock()
        return ConcreteBotEvaluator(mock_bot_class, "test_bot", mocker=mocker)

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.bot_name == "test_bot"
        assert evaluator.bot_class is not None

    def test_load_test_cases(self, evaluator):
        """Test loading test cases."""
        test_cases = evaluator.load_test_cases()

        assert len(test_cases) == 1
        assert test_cases[0].question == "Test question"
        assert test_cases[0].expected_output == "Test output"
        assert test_cases[0].expected_tools == ["test_tool"]

    def test_validate_test_case(self, evaluator):
        """Test test case validation."""
        valid_case = TestCaseSpec(question="Valid question")
        invalid_case = TestCaseSpec(question=None)

        assert evaluator.validate_test_case(valid_case) is True
        assert evaluator.validate_test_case(invalid_case) is False

    def test_create_bot_instance(self, evaluator):
        """Test bot instance creation."""
        test_case = TestCaseSpec(question="Test")
        bot = evaluator.create_bot_instance("gpt-4", test_case)

        assert bot is not None
        assert "configurable" in bot.config

    @pytest.mark.asyncio
    async def test_collect_response(self, evaluator, mocker):
        """Test response collection."""
        mock_bot = mocker.Mock()
        test_case = TestCaseSpec(question="Test question")

        response = await evaluator.collect_response(mock_bot, test_case)

        assert "messages" in response
        assert len(response["messages"]) == 4
        assert isinstance(response["messages"][0], HumanMessage)
        assert isinstance(response["messages"][3], AIMessage)

    def test_extract_tool_results(self, evaluator):
        """Test tool results extraction."""
        response = {
            "messages": [
                HumanMessage(content="User"),
                AIMessage(content="Tool"),
                AIMessage(content='{"results": [{"chunk_content": "Test"}]}'),
                AIMessage(content="Bot"),
            ]
        }
        test_case = TestCaseSpec(question="Test", expected_tools=["test_tool"])

        results = evaluator.extract_tool_results(response, test_case)
        assert len(results) == 1
        assert results[0]["chunk_content"] == "Test"

    def test_extract_tool_results_no_tools(self, evaluator):
        """Test tool results extraction when no tools expected."""
        response = {
            "messages": [HumanMessage(content="User"), AIMessage(content="Bot")]
        }
        test_case = TestCaseSpec(question="Test")

        results = evaluator.extract_tool_results(response, test_case)
        assert results == []

    def test_extract_tool_calls(self, evaluator, mocker):
        """Test tool calls extraction."""
        mock_tool_call = mocker.Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        response = {
            "messages": [
                HumanMessage(content="User"),
                AIMessage(
                    content="Tool call",
                    additional_kwargs={"tool_calls": [mock_tool_call]},
                ),
            ]
        }

        tool_calls = evaluator.extract_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "test_tool"
        assert tool_calls[0].input_parameters == {"arg": "value"}

    def test_extract_tool_calls_dict_format(self, evaluator):
        """Test tool calls extraction with dictionary format (from actual API)."""
        # This is the format returned by the actual LiteLLM/OpenAI API
        tool_call_dict = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg": "value"}',
            },
        }

        response = {
            "messages": [
                HumanMessage(content="User"),
                AIMessage(
                    content="Tool call",
                    additional_kwargs={"tool_calls": [tool_call_dict]},
                ),
            ]
        }

        tool_calls = evaluator.extract_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "test_tool"
        assert tool_calls[0].input_parameters == {"arg": "value"}

    def test_extract_tool_calls_no_tools(self, evaluator):
        """Test tool calls extraction with no tool calls."""
        response = {"messages": [HumanMessage(content="User message")]}

        tool_calls = evaluator.extract_tool_calls(response)
        assert tool_calls == []

    def test_extract_tool_results_different_position(self, evaluator):
        """Test tool results found when JSON message is not at index 2."""
        response = {
            "messages": [
                HumanMessage(content="User"),
                AIMessage(content="Thinking..."),
                AIMessage(content="Still thinking..."),
                AIMessage(content='{"results": [{"chunk_content": "Found it"}]}'),
                AIMessage(content="Final answer"),
            ]
        }
        test_case = TestCaseSpec(question="Test", expected_tools=["test_tool"])
        results = evaluator.extract_tool_results(response, test_case)
        assert len(results) == 1
        assert results[0]["chunk_content"] == "Found it"

    def test_extract_tool_calls_different_position(self, evaluator, mocker):
        """Test tool calls found when tool call message is not at index 1."""
        mock_tool_call = mocker.Mock()
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"q": "test"}'

        response = {
            "messages": [
                HumanMessage(content="User"),
                AIMessage(content="Let me think..."),
                AIMessage(
                    content="Calling tool",
                    additional_kwargs={"tool_calls": [mock_tool_call]},
                ),
                AIMessage(content="Result"),
            ]
        }
        tool_calls = evaluator.extract_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search"

    def test_extract_tool_results_single_message(self, evaluator):
        """Test tool results extraction with TutorBot-style single message."""
        response = {
            "messages": [AIMessage(content="Here is the answer to your question.")]
        }
        test_case = TestCaseSpec(question="Test", expected_tools=["tutor_tool"])
        results = evaluator.extract_tool_results(response, test_case)
        assert results == []

    def test_extract_tool_calls_single_message(self, evaluator):
        """Test tool calls extraction with TutorBot-style single message."""
        response = {"messages": [AIMessage(content="Here is the answer.")]}
        tool_calls = evaluator.extract_tool_calls(response)
        assert tool_calls == []

    def test_create_llm_test_case(self, evaluator, mocker):
        """Test LLM test case creation."""
        test_case = TestCaseSpec(
            question="Test question",
            expected_output="Expected output",
            expected_tools=["test_tool"],
        )

        # Create proper tool call mock
        tool_call_mock = mocker.Mock()
        tool_call_mock.function.name = "test_tool"
        tool_call_mock.function.arguments = '{"arg": "value"}'

        response = {
            "messages": [
                HumanMessage(content="User message"),
                AIMessage(
                    content="Tool call",
                    additional_kwargs={"tool_calls": [tool_call_mock]},
                ),
                AIMessage(content='{"results": [{"chunk_content": "Test content"}]}'),
                AIMessage(content="Bot response"),
            ]
        }

        llm_test_case = evaluator.create_llm_test_case(
            test_case, response, "gpt-4o", "default"
        )

        assert isinstance(llm_test_case, LLMTestCase)
        assert llm_test_case.name == "test_bot-gpt-4o-default"
        assert llm_test_case.input == "Test question"
        assert llm_test_case.actual_output == "Bot response"
        assert llm_test_case.expected_output == "Expected output"
        assert len(llm_test_case.tools_called) == 1
        assert len(llm_test_case.expected_tools) == 1
        assert llm_test_case.metadata["bot_name"] == "test_bot"
        assert llm_test_case.metadata["model"] == "gpt-4o"
        assert llm_test_case.metadata["prompt_label"] == "default"

    def test_create_llm_test_case_context(self, evaluator):
        """Context holds extra_context plus every non-empty tool message."""
        test_case = TestCaseSpec(question="Test question")
        response = {
            "messages": [
                HumanMessage(content="User message"),
                ToolMessage(content='{"results": ["a"]}', tool_call_id="1"),
                AIMessage(content="Not a tool output"),
                ToolMessage(content="", tool_call_id="2"),
                ToolMessage(content="Second tool output", tool_call_id="3"),
                AIMessage(content="Bot response"),
            ],
            "extra_context": ["Facts from the system prompt"],
        }

        llm_test_case = evaluator.create_llm_test_case(
            test_case, response, "gpt-4o", "default"
        )

        expected = [
            "Facts from the system prompt",
            '{"results": ["a"]}',
            "Second tool output",
        ]
        assert llm_test_case.context == expected
        assert llm_test_case.retrieval_context == expected

    def test_create_llm_test_case_no_context(self, evaluator):
        """A response with no tool calls or extra context has empty context."""
        response = {"messages": [AIMessage(content="Bot response")]}

        llm_test_case = evaluator.create_llm_test_case(
            TestCaseSpec(question="Q"), response, "gpt-4o"
        )

        assert llm_test_case.context == []
        assert llm_test_case.retrieval_context == []

    def test_create_llm_test_case_warns_on_empty_extraction(self, evaluator, caplog):
        """Test that warnings are logged when tools expected but extraction is empty."""
        import logging

        test_case = TestCaseSpec(
            question="Test",
            expected_output="Expected",
            expected_tools=["search_tool"],
        )
        response = {
            "messages": [
                HumanMessage(content="User message"),
                AIMessage(content="Direct answer with no tools"),
            ]
        }

        with caplog.at_level(logging.WARNING, logger="ai_chatbots.evaluation.base"):
            evaluator.create_llm_test_case(test_case, response, "gpt-4o", "default")

        assert "Empty tool results" in caplog.text
        assert "Empty tool calls" in caplog.text
        assert "search_tool" in caplog.text

    def test_create_llm_test_case_no_warning_without_expected_tools(
        self, evaluator, caplog
    ):
        """Test that no warning is logged when no tools were expected."""
        import logging

        test_case = TestCaseSpec(question="Test", expected_output="Expected")
        response = {
            "messages": [
                HumanMessage(content="User message"),
                AIMessage(content="Answer"),
            ]
        }

        with caplog.at_level(logging.WARNING, logger="ai_chatbots.evaluation.base"):
            evaluator.create_llm_test_case(test_case, response, "gpt-4o", "default")

        assert "Empty tool results" not in caplog.text
        assert "Empty tool calls" not in caplog.text

    @pytest.mark.asyncio
    async def test_evaluate_model(self, evaluator):
        """Test model evaluation."""
        test_cases = [
            TestCaseSpec(question="Test 1"),
            TestCaseSpec(question=None),  # Invalid case
            TestCaseSpec(question="Test 2"),
        ]

        llm_test_cases = await evaluator.evaluate_model("gpt-4o", test_cases)

        # Should only process valid test cases
        assert len(llm_test_cases) == 2
        assert all(isinstance(case, LLMTestCase) for case in llm_test_cases)


@pytest.fixture
def evaluator(mocker):
    """Create a concrete evaluator, for the module-level tests below."""
    return ConcreteBotEvaluator(mocker.Mock(), "test_bot", mocker=mocker)


# question, reference, metadata per data file
FILES = {
    "a.json": [("A0", "ref", {}), ("A1", None, {}), (None, "ref", {})],
    "b.json": [("B0", "ref", {}), ("B1", None, {})],
}


@pytest.mark.parametrize(
    ("require_reference", "expected"),
    [
        (
            False,
            [
                ("A0", "a.json#0"),
                ("A1", "a.json#1"),
                ("B0", "b.json#0"),
                ("B1", "b.json#1"),
            ],
        ),
        (True, [("A0", "a.json#0"), ("B0", "b.json#0")]),
    ],
)
def test_load_cases_pools_files_and_drops(mocker, require_reference, expected):
    """Cases from every file are pooled and tagged; dropped ones are counted."""
    evaluator = ConcreteBotEvaluator(
        mocker.Mock(), "test_bot", data_file="a.json,b.json"
    )
    evaluator.stdout = mocker.Mock()
    mocker.patch.object(
        evaluator,
        "load_test_cases",
        side_effect=lambda: [
            TestCaseSpec(question=q, expected_output=ref, metadata=dict(meta))
            for q, ref, meta in FILES[evaluator.data_file]
        ],
    )

    cases = evaluator.load_cases(require_reference=require_reference)

    assert [(c.question, c.metadata["case_id"]) for c in cases] == expected
    output = " ".join(str(c.args[0]) for c in evaluator.stdout.write.call_args_list)
    assert "Skipped 1 test_bot cases: invalid for this bot" in output
    assert ("Skipped 2 test_bot cases: no reference answer" in output) is (
        require_reference
    )


def test_sample_cases_is_repeatable():
    """The same seed picks the same cases, in order; another seed differs."""
    cases = list(range(100))
    first = sample_cases(cases, 30)
    assert first == sample_cases(cases, 30)
    assert first == sorted(first)
    assert len(first) == 30
    assert sample_cases(cases, 30, seed=SAMPLE_SEED + 1) != first
    assert sample_cases(cases, 0) == cases


def test_failed_cases_leave_out_judge_failures():
    """A judge failure doesn't drop the case's other measures from a comparison."""
    failures = RunFailures()
    failures.add("bot", "pairwise judge error", case_id="f#0")
    failures.add("bot", "judge error", case_id="f#1")
    failures.add("bot", "empty answer", case_id="f#2")
    assert failures.failed_cases("bot") == {("f#2", None)}


@pytest.mark.asyncio
async def test_evaluate_model_records_generation_failures(evaluator, mocker):
    """A case that fails to generate is recorded, not just dropped."""
    failures = RunFailures()
    mocker.patch.object(
        evaluator, "collect_response", side_effect=[RuntimeError("boom"), None]
    )
    mocker.patch.object(evaluator, "create_llm_test_case", return_value="answer")
    cases = [
        TestCaseSpec(question="Q1", metadata={"case_id": "f#0"}),
        TestCaseSpec(question="Q2", metadata={"case_id": "f#1"}),
    ]

    answers = await evaluator.evaluate_model("m", cases, failures=failures)

    assert answers == ["answer"]
    assert [(f.kind, f.case_id, f.model, f.detail) for f in failures.items] == [
        ("answer failed to generate", "f#0", "m", "boom")
    ]


@pytest.mark.parametrize(
    ("name", "content", "is_error"),
    [
        ("search_courses", '{"error": "An error occurred", "results": []}', True),
        ("search_content_files", '{"results": [], "metadata": {}}', False),
        (
            "search_support_articles",
            '{"results": [], "metadata": {"error": "An error occurred"}}',
            True,
        ),
        ("get_video_transcript_chunk", '{"error": "An error occurred"}', True),
        # A planted account error is what some account cases test
        ("get_enrollments", '{"error": "could not check"}', False),
        ("search_courses", "not json", False),
    ],
)
def test_create_llm_test_case_search_errors(evaluator, name, content, is_error):
    """Only search tools that returned their error result are recorded."""
    response = {
        "messages": [
            ToolMessage(content=content, name=name, tool_call_id="1"),
            AIMessage(content="Answer"),
        ],
        "blocked_account_calls": ["get_enrollments"],
    }
    case = evaluator.create_llm_test_case(
        TestCaseSpec(question="Q", metadata={"case_id": "f#0"}), response, "m"
    )
    assert case.metadata["search_errors"] == ([name] if is_error else [])
    assert case.metadata["case_id"] == "f#0"
    assert case.metadata["tool_call_count"] == 1
    assert case.metadata["blocked_account_calls"] == ["get_enrollments"]
