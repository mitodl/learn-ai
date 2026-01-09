"""Base evaluation framework for chatbot RAG evaluation."""

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
    metric_thresholds: dict[str, float]
    confident_api_key: Optional[str] = None


class BaseBotEvaluator(ABC):
    """Abstract base class for bot-specific evaluators."""

    def __init__(
        self,
        bot_class,
        bot_name: str,
        *,
        data_file: Optional[str] = None,
        stdout=None,
        error_log_file: Optional[str] = None,
    ):
        self.bot_class = bot_class
        self.bot_name = bot_name
        self.data_file = data_file or "test_json/rag_evaluation.json"
        self.stdout = stdout  # Optional output wrapper for logging
        self.error_log_file = error_log_file or "rag_evaluation_errors.log"

    def _log_error(
        self,
        error_msg: str,
        test_case: Optional[TestCaseSpec] = None,
        exception: Optional[Exception] = None,
    ):
        """Log error to file with timestamp and details."""
        try:
            timestamp = datetime.now(tz=UTC).isoformat()
            log_entry = f"\n{'='*80}\n"
            log_entry += f"[{timestamp}] Error in {self.bot_name}\n"

            if test_case:
                log_entry += f"Test case: {test_case.question[:100]}\n"
                if test_case.metadata:
                    log_entry += f"Metadata: {test_case.metadata}\n"

            log_entry += f"Error: {error_msg}\n"

            if exception:
                log_entry += f"\nFull traceback:\n{traceback.format_exc()}\n"

            log_entry += f"{'='*80}\n"

            # Append to log file
            log_path = Path(self.error_log_file)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as log_err:  # noqa: BLE001
            # Don't let logging errors crash the evaluation
            if self.stdout:
                self.stdout.write(f"Failed to write to error log: {log_err}")

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
        result = []
        for t in tool_calls:
            # Handle both dict format (from actual API) and object format (from mocks)
            if isinstance(t, dict):
                function_data = t.get("function", {})
                name = function_data.get("name", "")
                arguments = function_data.get("arguments", "{}")
            else:
                name = t.function.name
                arguments = t.function.arguments
            result.append(
                ToolCall(
                    input_parameters=json.loads(arguments),
                    name=name,
                )
            )
        return result

    async def evaluate_model(
        self,
        model: str,
        test_cases: list[TestCaseSpec],
        instructions: Optional[str] = None,
        prompt_label: str = "default",
        max_concurrent: int = 10,
    ) -> list[LLMTestCase]:
        """Evaluate a single model against all test cases with concurrent execution.

        Args:
            model: Model name to evaluate
            test_cases: List of test case specifications
            instructions: Optional custom instructions/prompt
            prompt_label: Label for the prompt being used
            max_concurrent: Maximum number of concurrent bot response generations
        """
        # Filter valid test cases upfront
        valid_test_cases = [tc for tc in test_cases if self.validate_test_case(tc)]

        if not valid_test_cases:
            return []

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_test_case(test_case: TestCaseSpec) -> Optional[LLMTestCase]:
            """Process a single test case with semaphore limiting."""
            async with semaphore:
                try:
                    chatbot = self.create_bot_instance(
                        model, test_case, instructions=instructions
                    )
                    response = await self.collect_response(chatbot, test_case)

                    # Log response in real-time if stdout is available
                    if self.stdout:
                        actual_output = (
                            response["messages"][-1].content
                            if response.get("messages")
                            else ""
                        )
                        truncated = (
                            actual_output[:100] if actual_output else "(no response)"
                        )
                        question = test_case.question[:50]
                        self.stdout.write(
                            f"Response for '{question}' ({prompt_label}): "
                            f"{truncated}..."
                        )

                    # Add prompt info to test case metadata for better tracking
                    return self.create_llm_test_case(
                        test_case, response, model, prompt_label
                    )
                except Exception as e:  # noqa: BLE001
                    error_msg = f"Error processing test case: {e!s}"

                    # Log to console
                    if self.stdout:
                        question_short = test_case.question[:50]
                        self.stdout.write(
                            f"Error processing test case '{question_short}': {e}"
                        )

                    # Log to file with full details
                    self._log_error(error_msg, test_case=test_case, exception=e)

                    return None

        # Execute all test cases concurrently with semaphore limiting
        results = await asyncio.gather(
            *[process_test_case(tc) for tc in valid_test_cases], return_exceptions=False
        )

        # Filter out None results from errors
        llm_test_cases = [r for r in results if r is not None]

        # Report success/failure counts
        success_count = len(llm_test_cases)
        failure_count = len(valid_test_cases) - success_count

        if self.stdout:
            total = len(valid_test_cases)
            self.stdout.write(
                f"Model {model} ({prompt_label}): {success_count}/{total} "
                f"test cases succeeded"
            )
            if failure_count > 0:
                self.stdout.write(
                    f"WARNING: {failure_count} test cases failed. "
                    f"See {self.error_log_file} for details."
                )

        return llm_test_cases
