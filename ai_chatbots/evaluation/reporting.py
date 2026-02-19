"""Reporting classes for RAG evaluation results."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional, TextIO

import pandas as pd
from deepeval.evaluate.types import EvaluationResult
from django.conf import settings
from django.core.management.base import OutputWrapper

# Constants
DEFAULT_PASS_THRESHOLD = 0.5  # Default threshold for metrics without explicit threshold


class DualOutputWrapper:
    """Wrapper that writes to both stdout and a file simultaneously."""

    def __init__(self, stdout: OutputWrapper, file_path: Optional[str] = None):
        self.stdout = stdout
        self.file_path = file_path
        self.file: Optional[TextIO] = None

        if file_path:
            file_path_obj = Path(file_path)
            self.file = file_path_obj.open("w", encoding="utf-8")

    def write(self, msg: str, style_func=None, ending=None) -> None:
        """Write message to both stdout and file."""
        # Write to stdout (preserves original behavior)
        self.stdout.write(msg, style_func=style_func, ending=ending)

        # Also write to file if specified
        if self.file:
            if ending is None:
                ending = "\n"

            # Remove style codes for file output (optional)
            clean_msg = msg
            if style_func:
                # Could strip ANSI codes here if needed
                clean_msg = msg

            self.file.write(clean_msg + ending)
            self.file.flush()

    def flush(self):
        """Flush both outputs."""
        self.stdout.flush() if hasattr(self.stdout, "flush") else None
        if self.file:
            self.file.flush()

    def close(self):
        """Close the file (but not stdout)."""
        if self.file and not self.file.closed:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EvaluationReporter:
    """Handles reporting of evaluation results."""

    def __init__(self, stdout: OutputWrapper):
        self.stdout = stdout

    def is_inverse_metric(self, metric_name: str) -> bool:
        """
        Determine if a metric is inverse (lower scores are better).

        Uses a settings-based approach to identify metrics where lower scores
        indicate better performance (e.g., hallucination metrics).

        Args:
            metric_name: The metric name

        Returns:
            True if lower scores are better, False if higher scores are better
        """
        return metric_name.lower() in settings.AI_INVERSE_METRICS

    def normalize_score_for_aggregation(self, metric: str, score: float) -> float:
        """
        Normalize scores for aggregation so that higher is always better.

        For most metrics, higher scores are better. However, for inverse metrics
        (like hallucination where lower scores are better), this function inverts
        the scores so all metrics follow the same "higher is better" convention
        when calculating averages.

        Args:
            metric: The metric name
            score: The original score (0.0 to 1.0)

        Returns:
            Normalized score where higher is always better
        """
        if self.is_inverse_metric(metric):
            return 1.0 - score
        return score

    def generate_report(  # noqa: PLR0913
        self,
        results: EvaluationResult,
        models: list[str],
        bot_names: list[str],
        *,
        use_prompts: bool = True,
        metric_thresholds: dict[str, float] | None = None,
        metric_weights: dict[str, float] | None = None,
        evaluation_model: str = "",
        metric_names: list[str] | None = None,
    ) -> None:
        """Generate a comprehensive evaluation report."""
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("RAG EVALUATION REPORT")
        self.stdout.write("=" * 80)

        # Run configuration manifest
        self.run_manifest(
            models,
            bot_names,
            evaluation_model=evaluation_model,
            metric_names=metric_names or [],
            metric_thresholds=metric_thresholds,
            use_prompts=use_prompts,
        )

        # Create DataFrame for easier analysis
        df = self.create_results_dataframe(results)

        # Generate report sections
        self.summarize_per_bot_model(df, models, bot_names, metric_thresholds)
        self.model_comparison(df)
        if use_prompts:
            self.prompt_comparison(df)
        self.overall_performance(df)
        self.composite_leaderboard(df, metric_weights=metric_weights)
        self.detailed_results(df, models, bot_names)

        self.stdout.write("\n" + "=" * 80)

    def create_results_dataframe(self, results: EvaluationResult) -> pd.DataFrame:
        """Create DataFrame from evaluation results."""
        data = [
            {
                "bot": test_result.additional_metadata["bot_name"],
                "model": test_result.additional_metadata["model"],
                "prompt_label": test_result.additional_metadata.get(
                    "prompt_label", "default"
                ),
                "test_case": test_result.name,
                "question": test_result.additional_metadata.get("question", "N/A"),
                "metric": metric_data.name,
                "score": metric_data.score,
                "success": metric_data.success,
                "reason": getattr(metric_data, "reason", ""),
            }
            for test_result in results.test_results
            for metric_data in test_result.metrics_data
        ]
        return pd.DataFrame(data)

    def summarize_per_bot_model(  # noqa: C901, PLR0912
        self,
        df: pd.DataFrame,
        models: list[str],
        bot_names: list[str],
        metric_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Summarize results per bot, model, and prompt."""
        self.stdout.write("\n📊 SUMMARY BY BOT, MODEL, PROMPT")
        self.stdout.write("-" * 50)

        if df.empty or not all(
            col in df.columns
            for col in ["bot", "model", "prompt_label", "metric", "score"]
        ):
            self.stdout.write("No data to summarize")
            return

        for bot in bot_names:
            if bot in df["bot"].values:
                self.stdout.write(f"\n🤖 {bot.upper()} BOT:")
                bot_df = df[df["bot"] == bot]

                has_results = False
                for model in models:
                    if model in bot_df["model"].values:
                        has_results = True
                        model_df = bot_df[bot_df["model"] == model]

                        # Get unique prompts for this bot/model combination
                        prompts = sorted(
                            model_df["prompt_label"].unique(),
                            key=lambda x: (x != "default", x),
                        )

                        self.stdout.write(f"\n  📱 Model: {model}")

                        for prompt in prompts:
                            prompt_df = model_df[model_df["prompt_label"] == prompt]
                            passes = prompt_df["success"].sum()
                            total = len(prompt_df)

                            prompt_msg = (
                                f"Prompt {prompt}"
                                if prompt != "default"
                                else "Default Prompt"
                            )
                            self.stdout.write(
                                f"\n    🎯 {prompt_msg} ({passes}/{total} metric checks passed)"  # noqa: E501
                            )

                            # Group by metric and get mean scores
                            metric_stats = prompt_df.groupby("metric").agg(
                                score=("score", "mean")
                            )

                            for metric, row in metric_stats.iterrows():
                                score = row["score"]

                                # Determine PASS/FAIL based on threshold
                                if metric_thresholds and metric in metric_thresholds:
                                    threshold = metric_thresholds[metric]
                                    # For inverse metrics, lower scores are better
                                    if self.is_inverse_metric(metric):
                                        overall_pass = score <= threshold
                                    else:
                                        overall_pass = score >= threshold
                                else:
                                    # Fallback: use default threshold
                                    overall_pass = score > DEFAULT_PASS_THRESHOLD

                                status = "✅ PASS" if overall_pass else "❌ FAIL"

                                self.stdout.write(
                                    f"      • {metric}: {score:.3f} {status}"
                                )

                if not has_results:
                    self.stdout.write("  No test cases defined for this bot")
            else:
                self.stdout.write(f"\n🤖 {bot.upper()} BOT:")
                self.stdout.write("  No test cases defined for this bot")

    def model_comparison(self, df: pd.DataFrame) -> None:
        """Compare models based on their average scores."""
        self.stdout.write("\n\n🔄 MODEL COMPARISON")
        self.stdout.write("-" * 50)

        if df.empty or not all(
            col in df.columns for col in ["model", "metric", "score"]
        ):
            self.stdout.write("No data to compare")
            return

        model_avg = (
            df.groupby(["model", "metric"])["score"].mean().unstack(fill_value=0)
        )

        for metric in model_avg.columns:
            self.stdout.write(f"\n📈 {metric}:")
            # For inverse metrics, lower scores are better, so sort ascending
            ascending_order = self.is_inverse_metric(metric)
            metric_scores = model_avg[metric].sort_values(ascending=ascending_order)
            for i, (model, score) in enumerate(metric_scores.items()):
                self.stdout.write(f"  {i + 1}. {model}: {score:.3f}")

    def prompt_comparison(self, df: pd.DataFrame) -> None:
        """Compare prompts based on their average scores."""
        self.stdout.write("\n\n🎯 PROMPT COMPARISON")
        self.stdout.write("-" * 50)

        if df.empty or not all(
            col in df.columns for col in ["prompt_label", "metric", "score"]
        ):
            self.stdout.write("No data to compare")
            return

        prompt_avg = (
            df.groupby(["prompt_label", "metric"])["score"].mean().unstack(fill_value=0)
        )

        for metric in prompt_avg.columns:
            self.stdout.write(f"\n📈 {metric}:")

            # Sort prompts: default first, then #1, #2, etc.
            def prompt_sort_key(prompt):
                if prompt == "default":
                    return (0, "")  # Default comes first
                elif prompt.startswith("#"):
                    try:
                        return (1, int(prompt[1:]))  # Extract number for sorting
                    except ValueError:
                        return (2, prompt)  # Fallback for unexpected format
                else:
                    return (2, prompt)  # Other prompts last

            sorted_prompts = sorted(prompt_avg[metric].index, key=prompt_sort_key)

            for i, prompt in enumerate(sorted_prompts):
                score = prompt_avg[metric][prompt]
                prompt_display = (
                    "Default Prompt" if prompt == "default" else f"Prompt {prompt}"
                )
                self.stdout.write(f"  {i + 1}. {prompt_display}: {score:.3f}")

    def overall_performance(self, df: pd.DataFrame) -> None:
        """Calculate and display overall performance of each model and prompt."""
        self.stdout.write("\n\n🏆 OVERALL PERFORMANCE")
        self.stdout.write("-" * 50)

        if df.empty or not all(col in df.columns for col in ["model", "score"]):
            self.stdout.write("No data for overall performance")
            return

        # Create a copy of the dataframe with normalized scores for aggregation
        df_normalized = df.copy()
        df_normalized["normalized_score"] = df_normalized.apply(
            lambda row: self.normalize_score_for_aggregation(
                row["metric"], row["score"]
            ),
            axis=1,
        )

        # Overall performance by model
        self.stdout.write("\n📱 BY MODEL:")
        overall_avg = (
            df_normalized.groupby("model")["normalized_score"]
            .mean()
            .sort_values(ascending=False)
        )

        for i, (model, avg_score) in enumerate(overall_avg.items()):
            self.stdout.write(f"  {i + 1}. {model}: {avg_score:.3f}")

        # Overall performance by prompt
        if "prompt_label" in df.columns:
            self.stdout.write("\n🎯 BY PROMPT:")
            prompt_avg = df_normalized.groupby("prompt_label")[
                "normalized_score"
            ].mean()

            # Sort prompts: default first, then #1, #2, etc.
            def prompt_sort_key(prompt):
                if prompt == "default":
                    return (0, "")  # Default comes first
                elif prompt.startswith("#"):
                    try:
                        return (1, int(prompt[1:]))  # Extract number for sorting
                    except ValueError:
                        return (2, prompt)  # Fallback for unexpected format
                else:
                    return (2, prompt)  # Other prompts last

            sorted_prompts = sorted(prompt_avg.index, key=prompt_sort_key)

            for i, prompt in enumerate(sorted_prompts):
                avg_score = prompt_avg[prompt]
                prompt_display = (
                    "Default Prompt" if prompt == "default" else f"Prompt {prompt}"
                )
                self.stdout.write(f"  {i + 1}. {prompt_display}: {avg_score:.3f}")

    def run_manifest(  # noqa: PLR0913
        self,
        models: list[str],
        bot_names: list[str],
        *,
        evaluation_model: str,
        metric_names: list[str],
        metric_thresholds: dict[str, float] | None = None,
        use_prompts: bool = True,
    ) -> None:
        """Display run configuration at the top of the report."""
        self.stdout.write("\n📝 RUN CONFIGURATION")
        self.stdout.write("-" * 50)
        self.stdout.write(
            f"  Timestamp: {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        self.stdout.write(f"  Models evaluated: {', '.join(models)}")
        if evaluation_model:
            self.stdout.write(f"  Judge model: {evaluation_model}")
        self.stdout.write(f"  Bots: {', '.join(bot_names)}")
        if metric_names:
            self.stdout.write(f"  Metrics: {', '.join(metric_names)}")
        if metric_thresholds:
            threshold_strs = [f"{k}={v}" for k, v in sorted(metric_thresholds.items())]
            self.stdout.write(f"  Thresholds: {', '.join(threshold_strs)}")
        self.stdout.write(f"  Prompt sweep: {'enabled' if use_prompts else 'disabled'}")

    def composite_leaderboard(
        self,
        df: pd.DataFrame,
        *,
        metric_weights: dict[str, float] | None = None,
    ) -> None:
        """Display a composite leaderboard with optional metric weights.

        Produces a ranked table of model+bot combinations. Default weights
        are equal across all metrics.
        """
        self.stdout.write("\n\n🏅 COMPOSITE LEADERBOARD")
        self.stdout.write("-" * 50)

        if df.empty or not all(
            col in df.columns for col in ["model", "bot", "metric", "score"]
        ):
            self.stdout.write("No data for composite leaderboard")
            return

        # Normalize scores (invert hallucination etc.)
        df_norm = df.copy()
        df_norm["normalized_score"] = df_norm.apply(
            lambda row: self.normalize_score_for_aggregation(
                row["metric"], row["score"]
            ),
            axis=1,
        )

        # Determine and normalize weights
        all_metrics = df_norm["metric"].unique()
        raw_weights = {m: (metric_weights or {}).get(m, 1.0) for m in all_metrics}
        weight_sum = sum(raw_weights.values())
        norm_weights = {m: w / weight_sum for m, w in raw_weights.items()}

        # Apply weights per row
        df_norm["weighted_score"] = df_norm.apply(
            lambda row: row["normalized_score"] * norm_weights.get(row["metric"], 0.0),
            axis=1,
        )

        # Composite score per model+bot: sum weighted scores, average across test cases
        metric_count = len(all_metrics)
        grouped = df_norm.groupby(["model", "bot"]).agg(
            total_weighted=("weighted_score", "sum"),
            metric_rows=("weighted_score", "count"),
        )
        grouped["n_test_cases"] = grouped["metric_rows"] / max(metric_count, 1)
        grouped["composite_score"] = grouped["total_weighted"] / grouped[
            "n_test_cases"
        ].clip(lower=1)
        composite = grouped.reset_index().sort_values(
            "composite_score", ascending=False
        )

        # Display per model+bot
        self.stdout.write("\nRanking by Model + Bot:")
        for rank, (_, row) in enumerate(composite.iterrows(), 1):
            self.stdout.write(
                f"  {rank}. {row['model']} / {row['bot']}: "
                f"{row['composite_score']:.3f}"
            )

        # Per-model aggregate
        model_composite = (
            composite.groupby("model")["composite_score"]
            .mean()
            .sort_values(ascending=False)
        )
        self.stdout.write("\nRanking by Model (avg across bots):")
        for rank, (model, score) in enumerate(model_composite.items(), 1):
            self.stdout.write(f"  {rank}. {model}: {score:.3f}")

        # Show weights
        if metric_weights:
            self.stdout.write("\nMetric weights used:")
            for m, w in sorted(norm_weights.items()):
                self.stdout.write(f"  {m}: {w:.3f}")
        else:
            self.stdout.write("\nMetric weights: equal (default)")

    def detailed_results(
        self, df: pd.DataFrame, models: list[str], bot_names: list[str]
    ) -> None:
        """Display detailed results for each bot, model, and prompt."""
        self.stdout.write("\n\n📋 DETAILED RESULTS")
        self.stdout.write("-" * 50)

        if df.empty or "bot" not in df.columns:
            self.stdout.write("No detailed results to display")
            return

        for bot in bot_names:
            bot_results = df[df["bot"] == bot]
            if not bot_results.empty:
                self.stdout.write(f"\n🤖 {bot.upper()} BOT DETAILS:")

                for model in models:
                    model_results = bot_results[bot_results["model"] == model]
                    if not model_results.empty:
                        self.stdout.write(f"\n  📱 Model: {model}")

                        # Get unique prompts for this bot/model combination
                        prompts = sorted(
                            model_results["prompt_label"].unique(),
                            key=lambda x: (x != "default", x),
                        )

                        for prompt in prompts:
                            prompt_results = model_results[
                                model_results["prompt_label"] == prompt
                            ]

                            prompt_display = (
                                "Default Prompt"
                                if prompt == "default"
                                else f"Prompt {prompt}"
                            )
                            self.stdout.write(f"\n    🎯 {prompt_display}:")

                            # Group by question to show results for each test case
                            for question in prompt_results["question"].unique():
                                question_results = prompt_results[
                                    prompt_results["question"] == question
                                ]

                                self.stdout.write(f"\n      ❓ Question: {question}")

                                for _, row in question_results.iterrows():
                                    status = "✅" if row["success"] else "❌"
                                    self.stdout.write(
                                        f"        {status} {row['metric']}: "
                                        f"{row['score']:.3f}"
                                    )

                                    if (
                                        not row["success"]
                                        and row["reason"]
                                        and len(str(row["reason"]).strip()) > 0
                                    ):
                                        self.stdout.write(
                                            f"           └─ {row['reason']}"
                                        )


class SummaryReporter:
    """Generates summary reports for evaluation results."""

    def __init__(self, stdout: OutputWrapper):
        self.stdout = stdout

    def generate_summary(self, results: EvaluationResult) -> None:
        """Generate a brief summary of evaluation results."""
        df = self.create_results_dataframe(results)

        if df.empty or "success" not in df.columns:
            self.stdout.write("\n📊 EVALUATION SUMMARY")
            self.stdout.write("-" * 40)
            self.stdout.write("No test results to summarize")
            return

        total_tests = len(df)
        passed_tests = df["success"].sum()
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        self.stdout.write("\n📊 EVALUATION SUMMARY")
        self.stdout.write(f"Total Tests: {total_tests}")
        self.stdout.write(f"Passed: {passed_tests}")
        self.stdout.write(f"Failed: {total_tests - passed_tests}")
        self.stdout.write(f"Pass Rate: {pass_rate:.1f}%")

    def create_results_dataframe(self, results: EvaluationResult) -> pd.DataFrame:
        """Create DataFrame from evaluation results."""
        data = [
            {
                "bot": test_result.additional_metadata["bot_name"],
                "model": test_result.additional_metadata["model"],
                "metric": metric_data.name,
                "success": metric_data.success,
            }
            for test_result in results.test_results
            for metric_data in test_result.metrics_data
        ]
        return pd.DataFrame(data)
