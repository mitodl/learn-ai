"""Orchestrator for running RAG evaluations across multiple bots and models."""

import os
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

import deepeval
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from django.core.management.base import OutputWrapper

from ai_chatbots.api import get_langsmith_prompt
from ai_chatbots.evaluation.base import EvaluationConfig
from ai_chatbots.evaluation.evaluators import BOT_EVALUATORS
from ai_chatbots.evaluation.reporting import EvaluationReporter
from ai_chatbots.evaluation.timeout_wrapper import wrap_metrics_with_timeout
from main.test_utils import load_json_with_settings

# Constants
ERROR_LOG_MIN_SIZE = 500  # Minimum file size to indicate errors beyond header


class EvaluationOrchestrator:
    """Orchestrates evaluation across multiple bots and models."""

    def __init__(self, stdout: OutputWrapper):
        self.stdout = stdout
        self.reporter = EvaluationReporter(stdout)

    def create_evaluation_config(
        self,
        models: list[str],
        evaluation_model: str,
        metric_thresholds: Optional[dict[str, float]] = None,
        timeout_seconds: int = 360,
    ) -> EvaluationConfig:
        """Create evaluation configuration with metrics."""
        if metric_thresholds is None:
            metric_thresholds = {
                "ContextualPrecision": 0.7,
                "ContextualRelevancy": 0.5,
                "ContextualRecall": 0.7,
                "Hallucination": 0.0,
                "AnswerRelevancy": 0.7,
                "Faithfulness": 0.7,
            }

        metrics = [
            ContextualPrecisionMetric(
                threshold=metric_thresholds["ContextualPrecision"],
                model=evaluation_model,
                include_reason=True,
            ),
            ContextualRelevancyMetric(
                threshold=metric_thresholds["ContextualRelevancy"],
                model=evaluation_model,
                include_reason=True,
            ),
            ContextualRecallMetric(
                threshold=metric_thresholds["ContextualRecall"],
                model=evaluation_model,
                include_reason=True,
            ),
            HallucinationMetric(
                threshold=metric_thresholds["Hallucination"],
                model=evaluation_model,
                include_reason=True,
            ),
            AnswerRelevancyMetric(
                threshold=metric_thresholds["AnswerRelevancy"],
                model=evaluation_model,
                include_reason=True,
            ),
        ]

        # Wrap metrics with timeout functionality
        timeout_wrapped_metrics = wrap_metrics_with_timeout(metrics, timeout_seconds)

        return EvaluationConfig(
            models=models,
            evaluation_model=evaluation_model,
            metrics=timeout_wrapped_metrics,
            metric_thresholds=metric_thresholds,
            confident_api_key=os.environ.get("CONFIDENT_AI_API_KEY"),
        )

    def _load_prompts_data(self, prompts_file: str) -> dict:
        """Load alternative prompts from file if enabled."""
        prompts_data = {}

        prompts_file = prompts_file or "test_json/rag_evaluation_prompts.json"
        try:
            prompts_data = load_json_with_settings(prompts_file)
            self.stdout.write(f"Loaded alternative prompts from {prompts_file}")
        except (FileNotFoundError, JSONDecodeError):
            self.stdout.write(f"Warning: {prompts_file} not found or invalid")
        return prompts_data

    async def _collect_and_evaluate_bot(  # noqa: C901, PLR0913
        self,
        bot_name: str,
        config: EvaluationConfig,
        *,
        data_file: Optional[str],
        use_prompts: Optional[bool],
        prompts_data: Optional[dict],
        max_concurrent: int,
        batch_size: int,
        error_log_file: str,
    ) -> EvaluationResult:
        """Collect test cases for a bot and evaluate them immediately.

        This processes one bot at a time, generating responses and evaluating them
        in batches to avoid memory issues from collecting all test cases upfront.
        """
        if bot_name not in BOT_EVALUATORS:
            self.stdout.write(f"Warning: Unknown bot '{bot_name}', skipping")
            return EvaluationResult(
                test_results=[], confident_link=None, test_run_id=None
            )

        bot_class, evaluator_class = BOT_EVALUATORS[bot_name]
        evaluator = evaluator_class(
            bot_class,
            bot_name,
            data_file=data_file,
            stdout=self.stdout,
            error_log_file=error_log_file,
        )

        # Load and validate test cases for this bot
        bot_test_cases = evaluator.load_test_cases()
        self.stdout.write(f"Loaded {len(bot_test_cases)} test cases for {bot_name}")

        # Get prompts for this bot (default + alternatives)
        bot_prompts = [None]  # Default prompt (None means use bot's default)
        if use_prompts and bot_name in prompts_data:
            bot_prompts.extend(prompts_data[bot_name])

        all_llm_test_cases = []
        batch_results = []

        # Evaluate each model with each prompt
        for model in config.models:
            for prompt in bot_prompts:
                # Resolve prompt details
                if prompt is None:
                    prompt_label = "default"
                    prompt_text = None
                else:
                    prompt_label = prompt["name"]
                    prompt_text = prompt.get("text") or get_langsmith_prompt(
                        prompt_label
                    )

                if prompt_label != "default" and not prompt_text:
                    self.stdout.write(f"No prompt text for '{prompt_label}', skipping")
                    continue

                self.stdout.write(
                    f"Evaluating {bot_name} with {model} using {prompt_label}"
                )

                try:
                    # Generate bot responses concurrently (this is now parallel!)
                    model_test_cases = await evaluator.evaluate_model(
                        model,
                        bot_test_cases,
                        instructions=prompt_text,
                        prompt_label=prompt_label,
                        max_concurrent=max_concurrent,
                    )
                    all_llm_test_cases.extend(model_test_cases)

                    # Process batches (loop until remaining < batch_size)
                    while batch_size > 0 and len(all_llm_test_cases) >= batch_size:
                        batch = all_llm_test_cases[:batch_size]
                        remaining = all_llm_test_cases[batch_size:]

                        batch_num = len(batch_results) + 1
                        self.stdout.write(
                            f"\n===== Evaluating batch {batch_num} for {bot_name} "
                            f"({len(batch)} test cases) ====="
                        )

                        batch_result = self._run_deepeval_evaluation(
                            batch, config, max_concurrent
                        )
                        batch_results.append(batch_result)

                        # Clear processed test cases to free memory
                        all_llm_test_cases = remaining

                        self.stdout.write(
                            f"Batch {batch_num} complete: "
                            f"{len(batch_result.test_results)} results"
                        )

                except Exception as e:  # noqa: BLE001
                    self.stdout.write(
                        f"Error on {bot_name} with {model} and {prompt_label}: {e}"
                    )
                    continue

        # Evaluate any remaining test cases
        if all_llm_test_cases:
            if batch_results:
                # We've been batching, so process the final batch
                batch_num = len(batch_results) + 1
                self.stdout.write(
                    f"\n===== Evaluating final batch {batch_num} for {bot_name} "
                    f"({len(all_llm_test_cases)} test cases) ====="
                )
            else:
                # No batching was done, evaluate all at once
                num_cases = len(all_llm_test_cases)
                self.stdout.write(
                    f"\nEvaluating all {num_cases} test cases for {bot_name}"
                )

            final_result = self._run_deepeval_evaluation(
                all_llm_test_cases, config, max_concurrent
            )
            batch_results.append(final_result)

        # Merge all batch results for this bot
        bot_result = self._merge_evaluation_results(batch_results)
        self.stdout.write(
            f"\nCompleted evaluation for {bot_name}: "
            f"{len(bot_result.test_results)} total results"
        )

        return bot_result

    def _run_deepeval_evaluation(
        self, test_cases: list, config: EvaluationConfig, max_concurrent: int
    ) -> EvaluationResult:
        """Run the actual DeepEval evaluation."""
        self.stdout.write(
            f"Running evaluation on {len(test_cases)} test cases "
            f"with max_concurrent={max_concurrent}"
        )

        if not test_cases:
            self.stdout.write("No test cases available - skipping evaluation")
            return EvaluationResult(
                test_results=[], confident_link=None, test_run_id=None
            )

        # Create AsyncConfig and log its settings
        async_config = AsyncConfig(
            max_concurrent=max_concurrent,
            throttle_value=0,  # No throttling by default
        )
        self.stdout.write(
            f"AsyncConfig: max_concurrent={async_config.max_concurrent}, "
            f"throttle_value={async_config.throttle_value}"
        )

        if max_concurrent < len(test_cases):
            self.stdout.write(
                f"NOTE: DeepEval may show 'Evaluating {len(test_cases)} "
                f"test case(s) in parallel' but will actually limit to "
                f"{max_concurrent} concurrent executions"
            )

        return deepeval.evaluate(
            test_cases=test_cases,
            metrics=config.metrics,
            error_config=ErrorConfig(ignore_errors=True),
            async_config=async_config,
        )

    def _merge_evaluation_results(
        self, results_list: list[EvaluationResult]
    ) -> EvaluationResult:
        """Merge multiple EvaluationResult objects into one."""
        if not results_list:
            return EvaluationResult(
                test_results=[], confident_link=None, test_run_id=None
            )

        if len(results_list) == 1:
            return results_list[0]

        # Concatenate all test_results
        all_test_results = []
        for result in results_list:
            all_test_results.extend(result.test_results)

        # Use the first non-None confident_link and test_run_id
        confident_link = next(
            (r.confident_link for r in results_list if r.confident_link), None
        )
        test_run_id = next((r.test_run_id for r in results_list if r.test_run_id), None)

        return EvaluationResult(
            test_results=all_test_results,
            confident_link=confident_link,
            test_run_id=test_run_id,
        )

    def _initialize_error_log(self, error_log_file: str):
        """Initialize error log file with header."""
        try:
            timestamp = datetime.now(tz=UTC).isoformat()
            log_path = Path(error_log_file)

            header = f"""
{'#'*80}
RAG EVALUATION ERROR LOG
Started: {timestamp}
{'#'*80}

"""
            with log_path.open("w", encoding="utf-8") as f:
                f.write(header)

            self.stdout.write(f"Error log initialized: {error_log_file}")
        except Exception as e:  # noqa: BLE001
            self.stdout.write(f"Warning: Failed to initialize error log: {e}")

    async def run_evaluation(  # noqa: PLR0913
        self,
        config: EvaluationConfig,
        *,
        bot_names: Optional[list[str]] = None,
        data_file: Optional[str] = None,
        use_prompts: Optional[bool] = True,
        prompts_file: Optional[str] = None,
        max_concurrent: Optional[int] = 10,
        batch_size: Optional[int] = 0,
        error_log_file: Optional[str] = None,
    ) -> EvaluationResult:
        """Run evaluation across specified bots and models.

        This processes one bot at a time, generating responses and evaluating them
        immediately to avoid memory issues. Results from all bots are accumulated
        and summarized at the end.

        Args:
            config: Evaluation configuration with models and metrics
            bot_names: List of bot names to evaluate (default: all bots)
            data_file: Path to custom data file with test cases
            use_prompts: Whether to use alternative prompts
            prompts_file: Path to prompts file
            max_concurrent: Maximum concurrent bot response generations
            batch_size: Number of test cases per batch (0 = no batching)
            error_log_file: Path to error log file (default: rag_evaluation_errors.log)
        """
        # Initialize error log file
        error_log_file = error_log_file or "rag_evaluation_errors.log"
        self._initialize_error_log(error_log_file)

        # Set up DeepEval authentication if API key is available
        if config.confident_api_key:
            deepeval.login(config.confident_api_key)

        # Determine which bots to evaluate
        bot_names = bot_names or list(BOT_EVALUATORS.keys())

        # Load alternative prompts if enabled
        prompts_data = self._load_prompts_data(prompts_file) if use_prompts else {}

        # Process each bot one at a time, accumulating results
        all_bot_results = []
        for bot_name in bot_names:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Processing bot: {bot_name}")
            self.stdout.write(f"{'='*60}")

            bot_result = await self._collect_and_evaluate_bot(
                bot_name,
                config,
                data_file=data_file,
                use_prompts=use_prompts,
                prompts_data=prompts_data,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                error_log_file=error_log_file,
            )
            all_bot_results.append(bot_result)

        # Merge results from all bots
        results = self._merge_evaluation_results(all_bot_results)

        # Log final summary
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("EVALUATION COMPLETE - FINAL SUMMARY")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(
            f"\nTotal test results across all bots: {len(results.test_results)}"
        )

        # Check if error log file has content (errors occurred)
        error_log_path = Path(error_log_file)
        if error_log_path.exists():
            file_size = error_log_path.stat().st_size
            if file_size > ERROR_LOG_MIN_SIZE:  # More than just the header
                self.stdout.write(
                    f"\n⚠️  ERRORS DETECTED - See {error_log_file} for details"
                )

        # Log test results breakdown by model and bot
        model_counts = {}
        bot_counts = {}
        for tr in results.test_results:
            model = tr.additional_metadata.get("model", "unknown")
            bot = tr.additional_metadata.get("bot_name", "unknown")
            metrics_count = len(tr.metrics_data) if tr.metrics_data else 0

            model_counts[model] = model_counts.get(model, 0) + 1
            bot_counts[bot] = bot_counts.get(bot, 0) + 1

            if metrics_count == 0:
                self.stdout.write(
                    f"WARNING: Test result for model {model} has no metrics_data"
                )

        self.stdout.write("\nResults by model:")
        for model, count in model_counts.items():
            self.stdout.write(f"  - {model}: {count} test results")

        self.stdout.write("\nResults by bot:")
        for bot, count in bot_counts.items():
            self.stdout.write(f"  - {bot}: {count} test results")

        # Generate final report
        self.reporter.generate_report(
            results,
            config.models,
            bot_names,
            metric_thresholds=config.metric_thresholds,
        )

        return results

    def get_available_bots(self) -> list[str]:
        """Get list of available bot names."""
        return list(BOT_EVALUATORS.keys())

    def validate_bot_names(self, bot_names: list[str]) -> list[str]:
        """Validate and filter bot names."""
        available_bots = self.get_available_bots()
        valid_bots = []

        for bot_name in bot_names:
            if bot_name in available_bots:
                valid_bots.append(bot_name)
            else:
                self.stdout.write(
                    f"Warning: Bot '{bot_name}' not found. "
                    f"Available bots: {', '.join(available_bots)}"
                )

        return valid_bots
