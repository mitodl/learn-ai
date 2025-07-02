"""Base evaluation framework for chatbot RAG evaluation."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from deepeval.test_case import LLMTestCase, ToolCall

# Constants for message parsing
MIN_MESSAGES_FOR_TOOL_RESULTS = 3
MIN_MESSAGES_FOR_TOOL_CALLS = 2


@dataclass
class TestCaseSpec:
    """Specification for a test case."""

    question: str
    expected_output: Optional[str] = None
    expected_tools: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "question": self.question,
            "expected_output": self.expected_output,
            "expected_tools": self.expected_tools or [],
            "metadata": self.metadata or {},
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""

    models: list[str]
    evaluation_model: str
    metrics: list[Any]
    confident_api_key: Optional[str] = None


class BaseBotEvaluator(ABC):
    """Abstract base class for bot-specific evaluators."""

    def __init__(self, bot_class, bot_name: str):
        self.bot_class = bot_class
        self.bot_name = bot_name

    @abstractmethod
    def load_test_cases(self) -> list[TestCaseSpec]:
        """Load test cases specific to this bot type."""

    @abstractmethod
    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate that a test case is properly formatted for this bot."""

    @abstractmethod
    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: Optional[str] = None
    ):
        """Create a bot instance configured for the given test case."""

    @abstractmethod
    async def collect_response(
        self, chatbot, test_case: TestCaseSpec
    ) -> dict[str, Any]:
        """Collect response from the bot for evaluation."""

    def create_llm_test_case(
        self,
        test_case: TestCaseSpec,
        response: dict[str, Any],
        model: str,
        prompt_label: str = "default",
    ) -> LLMTestCase:
        """Create DeepEval LLMTestCase from bot response."""
        # Extract tool information if available
        tool_results = self.extract_tool_results(response, test_case)
        retrieval_context = self.extract_retrieval_context(tool_results)
        tool_calls = self.extract_tool_calls(response)

        return LLMTestCase(
            name=f"{self.bot_name}-{model}-{prompt_label}",
            additional_metadata={
                "bot_name": self.bot_name,
                "model": model,
                "prompt_label": prompt_label,
                **test_case.to_dict(),
            },
            input=test_case.question,
            actual_output=response["messages"][-1].content,
            expected_output=test_case.expected_output,
            retrieval_context=retrieval_context,
            context=retrieval_context,
            tools_called=tool_calls,
            expected_tools=[ToolCall(name=t) for t in (test_case.expected_tools or [])],
        )

    def extract_tool_results(
        self, response: dict[str, Any], test_case: TestCaseSpec
    ) -> list[dict[str, Any]]:
        """Extract tool results from response."""
        if (
            not test_case.expected_tools
            or len(response["messages"]) < MIN_MESSAGES_FOR_TOOL_RESULTS
        ):
            return []

        try:
            tool_message = response["messages"][2]
            return json.loads(tool_message.content).get("results", [])
        except (json.JSONDecodeError, KeyError, IndexError):
            return []

    def extract_retrieval_context(
        self, tool_results: list[dict[str, Any]]
    ) -> list[str]:
        """Extract retrieval context from tool results."""
        if not tool_results:
            return []

        return [
            "\n\n".join(f.get("chunk_content", json.dumps(f)) for f in tool_results)
        ]

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from response."""
        if len(response["messages"]) < MIN_MESSAGES_FOR_TOOL_CALLS:
            return []

        tool_message = response["messages"][1]
        if not hasattr(tool_message, "additional_kwargs"):
            return []

        tool_calls = tool_message.additional_kwargs.get("tool_calls", [])
        return [
            ToolCall(
                input_parameters=json.loads(t.function.arguments),
                name=t.function.name,
            )
            for t in tool_calls
        ]

    async def evaluate_model(
        self,
        model: str,
        test_cases: list[TestCaseSpec],
        instructions: Optional[str] = None,
        prompt_label: str = "default",
    ) -> list[LLMTestCase]:
        """Evaluate a single model against all test cases."""
        llm_test_cases = []

        for test_case in test_cases:
            if not self.validate_test_case(test_case):
                continue

            chatbot = self.create_bot_instance(
                model, test_case, instructions=instructions
            )
            response = await self.collect_response(chatbot, test_case)

            # Add prompt info to test case metadata for better tracking
            llm_test_case = self.create_llm_test_case(
                test_case, response, model, prompt_label
            )
            llm_test_cases.append(llm_test_case)

        return llm_test_cases
