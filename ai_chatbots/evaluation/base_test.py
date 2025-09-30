"""Tests for evaluation.base module."""

from typing import Optional
from unittest.mock import Mock

import pytest
from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage, HumanMessage

from ai_chatbots.evaluation.base import (
    BaseBotEvaluator,
    EvaluationConfig,
    TestCaseSpec,
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
        config = EvaluationConfig(
            models=["gpt-4", "gpt-3.5"],
            evaluation_model="gpt-4",
            metrics=mock_metrics,
            confident_api_key="test-key",
        )

        assert config.models == ["gpt-4", "gpt-3.5"]
        assert config.evaluation_model == "gpt-4"
        assert config.metrics == mock_metrics
        assert config.confident_api_key == "test-key"

    def test_evaluation_config_optional_api_key(self):
        """Test EvaluationConfig without API key."""
        config = EvaluationConfig(
            models=["gpt-4"], evaluation_model="gpt-4", metrics=[]
        )

        assert config.confident_api_key is None


class ConcreteBotEvaluator(BaseBotEvaluator):
    """Concrete implementation for testing abstract methods."""

    def __init__(
        self, bot_class, bot_name: str, *, data_file: Optional[str] = None, mocker=None
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
        self, model: str, test_case: TestCaseSpec, instructions: Optional[str] = None
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

    def test_extract_retrieval_context(self, evaluator):
        """Test retrieval context extraction."""
        tool_results = [{"chunk_content": "Content 1"}, {"chunk_content": "Content 2"}]

        context = evaluator.extract_retrieval_context(tool_results)
        assert len(context) == 1
        assert "Content 1\n\nContent 2" in context[0]

    def test_extract_retrieval_context_empty(self, evaluator):
        """Test retrieval context extraction with empty results."""
        context = evaluator.extract_retrieval_context([])
        assert context == []

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

    def test_extract_tool_calls_no_tools(self, evaluator):
        """Test tool calls extraction with no tool calls."""
        response = {"messages": [HumanMessage(content="User message")]}

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
        assert llm_test_case.additional_metadata["bot_name"] == "test_bot"
        assert llm_test_case.additional_metadata["model"] == "gpt-4o"
        assert llm_test_case.additional_metadata["prompt_label"] == "default"

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
