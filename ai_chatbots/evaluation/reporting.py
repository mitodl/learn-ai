"""Reporting classes for RAG evaluation results."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import pandas as pd
from deepeval.evaluate.types import EvaluationResult
from django.core.management.base import OutputWrapper

from ai_chatbots.evaluation.base import judge_error

# Constants
DEFAULT_PASS_THRESHOLD = 0.5  # Default threshold for metrics without explicit threshold
NOT_JUDGED = "not judged"
QUESTION_WIDTH = 70


def _judged(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the rows the judge scored."""
    return df[df["error"].isna()] if "error" in df.columns else df


def _mean_scores(judged: pd.DataFrame, source: pd.DataFrame, by: str):
    """Mean score per `by` value and metric; NaN where nothing was judged."""
    return (
        judged.groupby([by, "metric"])["score"]
        .mean()
        .unstack()
        .reindex(
            index=sorted(source[by].unique()), columns=sorted(source["metric"].unique())
        )
    )


class DualOutputWrapper:
    """Wrapper that writes to both stdout and a file simultaneously."""

    def __init__(self, stdout: OutputWrapper, file_path: str | None = None):
        self.stdout = stdout
        self.file_path = file_path
        self.file: TextIO | None = None

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
        failures=None,
        answer_records: list[dict] | None = None,
        run_info: dict | None = None,
    ) -> None:
        """Generate a comprehensive evaluation report.

        Judge errors are listed with the run's other failures and left out of
        every score, mean and pass count.
        """
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("RAG EVALUATION REPORT")
        self.stdout.write("=" * 80)
        self.incomplete_banner(failures)

        # Run configuration manifest
        self.run_manifest(
            models,
            bot_names,
            evaluation_model=evaluation_model,
            metric_names=metric_names or [],
            metric_thresholds=metric_thresholds,
            use_prompts=use_prompts,
            run_info=run_info,
        )

        # Create DataFrame for easier analysis
        df = self.create_results_dataframe(results)
        judged = _judged(df)

        # Generate report sections
        self.summarize_per_bot_model(df, models, bot_names, metric_thresholds)
        self.model_comparison(judged, all_metrics=df)
        if use_prompts:
            self.prompt_comparison(judged, all_metrics=df)
        self.overall_performance(judged)
        self.composite_leaderboard(judged, metric_weights=metric_weights)
        self.detailed_results(df, models, bot_names)
        if answer_records:
            self.code_checks(answer_records)
        if failures:
            self.failures_section(failures)

        self.stdout.write("\n" + "=" * 80)

    def incomplete_banner(self, failures) -> None:
        """Say loudly that the run failed to measure something."""
        if not failures:
            return
        self.stdout.write(
            f"\n⚠️  INCOMPLETE RUN: {len(failures.items)} things could not be "
            "measured. See FAILURES; scores leave them out."
        )

    def failures_section(self, failures) -> None:
        """List everything the run failed to measure."""
        self.stdout.write("\n\n⚠️  FAILURES")
        self.stdout.write("-" * 50)
        counts: dict[tuple, int] = {}
        for failure in failures.items:
            key = (failure.bot, failure.model or "-", failure.kind)
            counts[key] = counts.get(key, 0) + 1
        for (bot, model, kind), count in sorted(counts.items()):
            self.stdout.write(f"  {bot} / {model}: {kind}: {count}")
        self.stdout.write("")
        for failure in failures.items:
            parts = [failure.bot, failure.model or "", failure.prompt_label or ""]
            where = " / ".join(p for p in parts if p)
            detail = f" ({failure.detail})" if failure.detail else ""
            self.stdout.write(
                f"  - {where}: {failure.kind}{detail}: {failure.case_id or ''} "
                f"{(failure.question or '')[:QUESTION_WIDTH]}"
            )

    def code_checks(self, answer_records: list[dict]) -> None:
        """Count, per bot, model and prompt, the answers each code check failed."""
        self.stdout.write("\n\n🔎 CODE CHECKS")
        self.stdout.write("-" * 50)
        groups: dict[tuple, list[dict]] = {}
        for record in answer_records:
            key = (record["bot"], record["model"], record["prompt_label"])
            groups.setdefault(key, []).append(record)
        for (bot, model, prompt_label), records in groups.items():
            checks = sorted({name for r in records for name in r["checks"]})
            failed = ", ".join(
                f"{name} {sum(bool(r['checks'].get(name)) for r in records)}"
                for name in checks
            )
            no_tool = sum(1 for r in records if not r.get("tool_call_count"))
            label = "" if prompt_label == "default" else f" ({prompt_label})"
            self.stdout.write(
                f"  {bot} / {model}{label}: {len(records)} answers; "
                f"failed: {failed or 'no checks'}; no tool call: {no_tool}"
            )

    def create_results_dataframe(self, results: EvaluationResult) -> pd.DataFrame:
        """Create DataFrame from evaluation results."""
        data = [
            {
                "bot": test_result.metadata["bot_name"],
                "model": test_result.metadata["model"],
                "prompt_label": test_result.metadata.get("prompt_label", "default"),
                "test_case": test_result.name,
                "question": test_result.metadata.get("question", "N/A"),
                "case_id": test_result.metadata.get("case_id"),
                "metric": metric_data.name,
                "score": None if judge_error(metric_data) else metric_data.score,
                "success": metric_data.success,
                "reason": getattr(metric_data, "reason", ""),
                "error": judge_error(metric_data),
            }
            for test_result in results.test_results
            for metric_data in test_result.metrics_data
        ]
        return pd.DataFrame(data)

    def summarize_per_bot_model(  # noqa: C901
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
                            all_rows = model_df[model_df["prompt_label"] == prompt]
                            prompt_df = _judged(all_rows)
                            passes = int(prompt_df["success"].sum())
                            total = len(prompt_df)
                            errors = len(all_rows) - total

                            prompt_msg = (
                                f"Prompt {prompt}"
                                if prompt != "default"
                                else "Default Prompt"
                            )
                            error_msg = f", {errors} judge errors" if errors else ""
                            self.stdout.write(
                                f"\n    🎯 {prompt_msg} ({passes}/{total} judged "
                                f"metric checks passed{error_msg})"
                            )

                            # Group by metric and get mean scores
                            metric_stats = prompt_df.groupby("metric").agg(
                                score=("score", "mean")
                            )
                            for metric in sorted(
                                set(all_rows["metric"]) - set(metric_stats.index)
                            ):
                                self.stdout.write(f"      • {metric}: {NOT_JUDGED}")

                            for metric, row in metric_stats.iterrows():
                                score = row["score"]

                                # Determine PASS/FAIL based on threshold
                                if metric_thresholds and metric in metric_thresholds:
                                    threshold = metric_thresholds[metric]
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

    def model_comparison(
        self, df: pd.DataFrame, *, all_metrics: pd.DataFrame | None = None
    ) -> None:
        """Compare models' mean scores; one with nothing judged is not ranked."""
        self.stdout.write("\n\n🔄 MODEL COMPARISON")
        self.stdout.write("-" * 50)

        source = df if all_metrics is None else all_metrics
        if source.empty or not all(
            col in source.columns for col in ["model", "metric", "score"]
        ):
            self.stdout.write("No data to compare")
            return

        model_avg = _mean_scores(df, source, "model")

        for metric in model_avg.columns:
            self.stdout.write(f"\n📈 {metric}:")
            metric_scores = model_avg[metric]
            ranked = metric_scores.dropna().sort_values(ascending=False)
            for i, (model, score) in enumerate(ranked.items()):
                self.stdout.write(f"  {i + 1}. {model}: {score:.3f}")
            for model in metric_scores[metric_scores.isna()].index:
                self.stdout.write(f"  -  {model}: {NOT_JUDGED}")

    def prompt_comparison(
        self, df: pd.DataFrame, *, all_metrics: pd.DataFrame | None = None
    ) -> None:
        """Compare prompts' mean scores; one with nothing judged says so."""
        self.stdout.write("\n\n🎯 PROMPT COMPARISON")
        self.stdout.write("-" * 50)

        source = df if all_metrics is None else all_metrics
        if source.empty or not all(
            col in source.columns for col in ["prompt_label", "metric", "score"]
        ):
            self.stdout.write("No data to compare")
            return

        prompt_avg = _mean_scores(df, source, "prompt_label")

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
                shown = NOT_JUDGED if pd.isna(score) else f"{score:.3f}"
                self.stdout.write(f"  {i + 1}. {prompt_display}: {shown}")

    def overall_performance(self, df: pd.DataFrame) -> None:
        """Calculate and display overall performance of each model and prompt."""
        self.stdout.write("\n\n🏆 OVERALL PERFORMANCE")
        self.stdout.write("-" * 50)

        if df.empty or not all(col in df.columns for col in ["model", "score"]):
            self.stdout.write("No data for overall performance")
            return

        # Overall performance by model
        self.stdout.write("\n📱 BY MODEL:")
        overall_avg = df.groupby("model")["score"].mean().sort_values(ascending=False)

        for i, (model, avg_score) in enumerate(overall_avg.items()):
            self.stdout.write(f"  {i + 1}. {model}: {avg_score:.3f}")

        # Overall performance by prompt
        if "prompt_label" in df.columns:
            self.stdout.write("\n🎯 BY PROMPT:")
            prompt_avg = df.groupby("prompt_label")["score"].mean()

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
        run_info: dict | None = None,
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
        for label, value in (run_info or {}).items():
            self.stdout.write(f"  {label}: {value}")

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

        # Determine and normalize weights
        all_metrics = df["metric"].unique()
        raw_weights = {m: (metric_weights or {}).get(m, 1.0) for m in all_metrics}
        weight_sum = sum(raw_weights.values())
        norm_weights = {m: w / weight_sum for m, w in raw_weights.items()}

        # Composite score per model+bot: the weighted mean of its per-metric mean
        # scores, with the weights renormalized over the metrics it was judged
        # on, so a metric the judge failed on doesn't count as a zero
        means = df.groupby(["model", "bot", "metric"])["score"].mean().reset_index()
        means["weight"] = means["metric"].map(norm_weights).fillna(0.0)
        means["weighted"] = means["score"] * means["weight"]
        grouped = means.groupby(["model", "bot"]).agg(
            total_weighted=("weighted", "sum"),
            total_weight=("weight", "sum"),
        )
        grouped["composite_score"] = grouped["total_weighted"] / grouped[
            "total_weight"
        ].where(grouped["total_weight"] > 0)
        composite = (
            grouped.reset_index()
            .dropna(subset=["composite_score"])
            .sort_values("composite_score", ascending=False)
        )

        # Display per model+bot
        self.stdout.write("\nRanking by Model + Bot:")
        for rank, (_, row) in enumerate(composite.iterrows(), 1):
            self.stdout.write(
                f"  {rank}. {row['model']} / {row['bot']}: {row['composite_score']:.3f}"
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

    def detailed_results(  # noqa: C901
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
                                    error = row.get("error")
                                    if isinstance(error, str):
                                        self.stdout.write(
                                            f"        ⚠️ {row['metric']}: judge "
                                            f"error: {error}"
                                        )
                                        continue
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
                "bot": test_result.metadata["bot_name"],
                "model": test_result.metadata["model"],
                "metric": metric_data.name,
                "success": metric_data.success,
            }
            for test_result in results.test_results
            for metric_data in test_result.metrics_data
        ]
        return pd.DataFrame(data)
