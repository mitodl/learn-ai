"""Base evaluation framework for chatbot RAG evaluation."""

import asyncio
import json
import logging
import math
import random
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asgiref.sync import sync_to_async
from deepeval.test_case import LLMTestCase, ToolCall

log = logging.getLogger(__name__)

# Fixed, so two runs of the same command judge the same cases
SAMPLE_SEED = 20261110

# A failed search here makes a run incomplete.  Not the account tools: some
# account cases plant a failing record, and "couldn't check" is what they test.
SEARCH_TOOLS = frozenset(
    {
        "search_courses",
        "search_content_files",
        "search_related_course_content_files",
        "get_video_transcript_chunk",
        "search_support_articles",
    }
)


def search_error(content: Any) -> str | None:
    """Return the error a search tool reported in its output, if any."""
    try:
        parsed = json.loads(content) if isinstance(content, str) else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    metadata = parsed.get("metadata")
    error = parsed.get("error") or (
        metadata.get("error") if isinstance(metadata, dict) else None
    )
    return str(error) if error else None


def judge_error(metric_data) -> str | None:
    """Return why the judge failed to score a deepeval metric, if it failed."""
    error = getattr(metric_data, "error", None)
    if isinstance(error, str) and error:
        return error
    score = getattr(metric_data, "score", None)
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "no score"
    return None


@dataclass
class Failure:
    """Something a run failed to measure."""

    bot: str
    kind: str
    question: str = ""
    case_id: str | None = None
    model: str | None = None
    prompt_label: str | None = None
    detail: str = ""


@dataclass
class RunFailures:
    """Everything a run failed to measure, so none of it is dropped quietly."""

    items: list[Failure] = field(default_factory=list)

    def add(self, bot: str, kind: str, case: LLMTestCase | None = None, **kwargs):
        """Record one failure, of the given answer (an LLMTestCase) if any."""
        if case is not None:
            metadata = case.metadata or {}
            kwargs = {
                "question": case.input,
                "case_id": metadata.get("case_id"),
                "model": metadata.get("model"),
                "prompt_label": metadata.get("prompt_label"),
                **kwargs,
            }
        self.items.append(Failure(bot=bot, kind=kind, **kwargs))

    def for_bot(self, bot: str) -> list[Failure]:
        """Return the failures of one bot."""
        return [f for f in self.items if f.bot == bot]

    def failed_cases(self, bot: str) -> set[tuple[str | None, str | None]]:
        """Return the (case_id, prompt_label) pairs of a bot with any failure."""
        return {(f.case_id, f.prompt_label) for f in self.for_bot(bot)}

    def __bool__(self) -> bool:
        return bool(self.items)


def sample_cases(cases: list, n: int | None, seed: int = SAMPLE_SEED) -> list:
    """Return a seeded sample of n cases in their order (all if n is 0 or None)."""
    if not n or n >= len(cases):
        return list(cases)
    picked = sorted(random.Random(seed).sample(range(len(cases)), n))  # noqa: S311
    return [cases[i] for i in picked]


@dataclass
class TestCaseSpec:
    """Specification for a test case."""

    question: str
    expected_output: str | None = None
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] | None = None

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
    confident_api_key: str | None = None
    # Applied to every judge call, including --pairwise
    timeout_seconds: int = 360
    max_retries: int = 3
    retry_delay: float = 5.0
    # Correctness against the reference answer, plus Faithfulness to the
    # retrieved material for the bots both sides search the same way
    correctness_only: bool = False
    # No deepeval metrics at all; only the pairwise judge
    pairwise_only: bool = False


class BaseBotEvaluator(ABC):
    """Abstract base class for bot-specific evaluators."""

    def __init__(
        self,
        bot_class,
        bot_name: str,
        *,
        data_file: str | list[str] | None = None,
        stdout=None,
        error_log_file: str | None = None,
    ):
        self.bot_class = bot_class
        self.bot_name = bot_name
        if isinstance(data_file, str):
            data_file = [f for f in data_file.split(",") if f]
        self.data_files = data_file or ["test_json/rag_evaluation.json"]
        self.data_file = self.data_files[0]
        self.stdout = stdout  # Optional output wrapper for logging
        self.error_log_file = error_log_file or "rag_evaluation_errors.log"

    def _log_error(
        self,
        error_msg: str,
        test_case: TestCaseSpec | None = None,
        exception: Exception | None = None,
    ):
        """Log error to file with timestamp and details."""
        try:
            timestamp = datetime.now(tz=UTC).isoformat()
            log_entry = f"\n{'=' * 80}\n"
            log_entry += f"[{timestamp}] Error in {self.bot_name}\n"

            if test_case:
                log_entry += f"Test case: {test_case.question[:100]}\n"
                if test_case.metadata:
                    log_entry += f"Metadata: {test_case.metadata}\n"

            log_entry += f"Error: {error_msg}\n"

            if exception:
                log_entry += f"\nFull traceback:\n{traceback.format_exc()}\n"

            log_entry += f"{'=' * 80}\n"

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
        """Load test cases specific to this bot type from self.data_file."""

    def load_cases(self, *, require_reference: bool = False) -> list[TestCaseSpec]:
        """Load every data file's valid cases, tagged "<file>#<index>"."""
        cases = []
        dropped: dict[str, int] = {}
        for data_file in self.data_files:
            self.data_file = data_file
            for index, case in enumerate(self.load_test_cases()):
                case.metadata = case.metadata or {}
                case.metadata["case_id"] = f"{Path(data_file).name}#{index}"
                if not self.validate_test_case(case):
                    reason = "invalid for this bot"
                elif require_reference and not case.expected_output:
                    reason = "no reference answer"
                else:
                    cases.append(case)
                    continue
                dropped[reason] = dropped.get(reason, 0) + 1
        self.data_file = self.data_files[0]
        if self.stdout:
            for reason, count in dropped.items():
                self.stdout.write(f"Skipped {count} {self.bot_name} cases: {reason}")
        return cases

    @abstractmethod
    def validate_test_case(self, test_case: TestCaseSpec) -> bool:
        """Validate that a test case is properly formatted for this bot."""

    @abstractmethod
    def create_bot_instance(
        self, model: str, test_case: TestCaseSpec, instructions: str | None = None
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
        tool_calls = self.extract_tool_calls(response)

        if test_case.expected_tools:
            if not tool_results:
                log.warning(
                    "Empty tool results for %s (model=%s, prompt=%s) "
                    "despite expected_tools=%s. "
                    "Retrieval context metrics will evaluate against empty context.",
                    self.bot_name,
                    model,
                    prompt_label,
                    test_case.expected_tools,
                )
            if not tool_calls:
                log.warning(
                    "Empty tool calls for %s (model=%s, prompt=%s) "
                    "despite expected_tools=%s.",
                    self.bot_name,
                    model,
                    prompt_label,
                    test_case.expected_tools,
                )

        tool_messages = [
            m
            for m in response.get("messages", [])
            if getattr(m, "type", None) == "tool"
        ]
        # Every tool output, so the metrics see all evidence the bot had
        context = response.get("extra_context", []) + [
            m.content for m in tool_messages if m.content
        ]
        search_errors = [
            m.name
            for m in tool_messages
            if getattr(m, "name", None) in SEARCH_TOOLS and search_error(m.content)
        ]
        log.info(
            "Evidence context for %s/%s %r: %d items, %d chars",
            self.bot_name,
            model,
            test_case.question[:50],
            len(context),
            sum(len(c) for c in context),
        )
        return LLMTestCase(
            name=f"{self.bot_name}-{model}-{prompt_label}",
            metadata={
                "bot_name": self.bot_name,
                "model": model,
                "prompt_label": prompt_label,
                **test_case.to_dict(),
                "case_id": (test_case.metadata or {}).get("case_id"),
                "search_errors": search_errors,
                "tool_call_count": len(tool_messages),
                "blocked_account_calls": response.get("blocked_account_calls", []),
            },
            input=test_case.question,
            actual_output=response["messages"][-1].content,
            expected_output=test_case.expected_output,
            retrieval_context=context,
            context=context,
            tools_called=tool_calls,
            expected_tools=[ToolCall(name=t) for t in (test_case.expected_tools or [])],
        )

    def extract_tool_results(
        self, response: dict[str, Any], test_case: TestCaseSpec
    ) -> list[dict[str, Any]]:
        """Extract tool results by scanning for JSON with 'results' key."""
        if not test_case.expected_tools:
            return []

        for message in response.get("messages", []):
            content = getattr(message, "content", None)
            if not content or not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "results" in parsed:
                    return parsed["results"]
            except (json.JSONDecodeError, TypeError):
                continue

        return []

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls by scanning for message with tool_calls."""
        for message in response.get("messages", []):
            if not hasattr(message, "additional_kwargs"):
                continue
            tool_calls_data = message.additional_kwargs.get("tool_calls", [])
            if not tool_calls_data:
                continue

            result = []
            for t in tool_calls_data:
                # Handle both dict (API) and object (mock) formats
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

        return []

    async def evaluate_model(  # noqa: PLR0913
        self,
        model: str,
        test_cases: list[TestCaseSpec],
        instructions: str | None = None,
        prompt_label: str = "default",
        max_concurrent: int = 10,
        failures: RunFailures | None = None,
    ) -> list[LLMTestCase]:
        """Evaluate a single model against all test cases with concurrent execution.

        Args:
            model: Model name to evaluate
            test_cases: List of test case specifications
            instructions: Optional custom instructions/prompt
            prompt_label: Label for the prompt being used
            max_concurrent: Maximum number of concurrent bot response generations
            failures: If given, every case that failed to generate is added to it
        """
        # Filter valid test cases upfront
        valid_test_cases = [tc for tc in test_cases if self.validate_test_case(tc)]

        if not valid_test_cases:
            return []

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_test_case(test_case: TestCaseSpec) -> LLMTestCase | None:
            """Process a single test case with semaphore limiting."""
            async with semaphore:
                try:
                    chatbot = await sync_to_async(self.create_bot_instance)(
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
                    if failures is not None:
                        failures.add(
                            self.bot_name,
                            "answer failed to generate",
                            question=test_case.question,
                            case_id=(test_case.metadata or {}).get("case_id"),
                            model=model,
                            prompt_label=prompt_label,
                            detail=str(e),
                        )

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
