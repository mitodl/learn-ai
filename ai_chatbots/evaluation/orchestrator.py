"""Orchestrator for running RAG evaluations across multiple bots and models."""

import os
from typing import Optional

import deepeval
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from django.core.management.base import OutputWrapper

from .base import EvaluationConfig
from .evaluators import BOT_EVALUATORS
from .reporting import EvaluationReporter


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
    ) -> EvaluationConfig:
        """Create evaluation configuration with metrics."""
        if metric_thresholds is None:
            metric_thresholds = {
                "ContextualRelevancy": 0.5,
                "ContextualRecall": 0.7,
                "Hallucination": 0.0,
                "AnswerRelevancy": 0.7,
                "Faithfulness": 0.7,
            }

        metrics = [
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
            FaithfulnessMetric(
                threshold=metric_thresholds["Faithfulness"],
                model=evaluation_model,
                include_reason=True,
            ),
        ]

        return EvaluationConfig(
            models=models,
            evaluation_model=evaluation_model,
            metrics=metrics,
            confident_api_key=os.environ.get("CONFIDENT_AI_API_KEY"),
        )

    async def run_evaluation(
        self,
        config: EvaluationConfig,
        bot_names: Optional[list[str]] = None,
    ) -> EvaluationResult:
        """Run evaluation across specified bots and models."""
        # Set up DeepEval logging if API key is available
        if config.confident_api_key:
            deepeval.login_with_confident_api_key(config.confident_api_key)

        # Determine which bots to evaluate
        if not bot_names:
            bot_names = list(BOT_EVALUATORS.keys())

        self.stdout.write(
            f"Evaluating bots: {', '.join(bot_names)}, "
            f"models: {', '.join(config.models)}"
        )

        # Collect all test cases
        test_cases = []
        for bot_name in bot_names:
            if bot_name not in BOT_EVALUATORS:
                self.stdout.write(f"Warning: Unknown bot '{bot_name}', skipping")
                continue

            bot_class, evaluator_class = BOT_EVALUATORS[bot_name]
            evaluator = evaluator_class(bot_class, bot_name)

            # Load and validate test cases for this bot
            bot_test_cases = evaluator.load_test_cases()
            self.stdout.write(f"Loaded {len(bot_test_cases)} test cases for {bot_name}")

            # Evaluate each model
            for model in config.models:
                self.stdout.write(f"Evaluating {bot_name} with {model}")

                try:
                    model_test_cases = await evaluator.evaluate_model(
                        model, bot_test_cases
                    )
                    test_cases.extend(model_test_cases)

                    # Log responses for debugging
                    for test_case in model_test_cases:
                        self.stdout.write(
                            f"Response for '{test_case.input}': "
                            f"{test_case.actual_output[:100]}..."
                        )

                except Exception as e:  # noqa: BLE001
                    self.stdout.write(f"Error evaluating {bot_name} with {model}: {e}")
                    continue

        # Run DeepEval evaluation
        self.stdout.write(f"Running evaluation on {len(test_cases)} test cases")
        results = deepeval.evaluate(test_cases=test_cases, metrics=config.metrics)

        # Generate report
        self.reporter.generate_report(results, config.models, bot_names)

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
