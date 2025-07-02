"""Reporting classes for RAG evaluation results."""

import pandas as pd
from deepeval.evaluate.types import EvaluationResult
from django.core.management.base import OutputWrapper


class EvaluationReporter:
    """Handles reporting of evaluation results."""

    def __init__(self, stdout: OutputWrapper):
        self.stdout = stdout

    def generate_report(
        self,
        results: EvaluationResult,
        models: list[str],
        bot_names: list[str],
        *,
        use_prompts: bool = True,
    ) -> None:
        """Generate a comprehensive evaluation report."""
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("RAG EVALUATION REPORT")
        self.stdout.write("=" * 80)

        # Create DataFrame for easier analysis
        df = self.create_results_dataframe(results)

        # Generate report sections
        self.summarize_per_bot_model(df, models, bot_names)
        self.model_comparison(df)
        if use_prompts:
            self.prompt_comparison(df)
        self.overall_performance(df)
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

    def summarize_per_bot_model(
        self, df: pd.DataFrame, models: list[str], bot_names: list[str]
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

                for model in models:
                    if model in bot_df["model"].values:
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
                                f"\n    🎯 {prompt_msg} ({passes}/{total} tests passed)"
                            )

                            # Group by metric and get mean scores
                            metric_scores = prompt_df.groupby("metric")["score"].mean()

                            for metric, score in metric_scores.items():
                                metric_result = prompt_df[prompt_df["metric"] == metric]
                                if not metric_result.empty:
                                    actual_success = metric_result.iloc[0]["success"]
                                    reason = metric_result.iloc[0]["reason"]
                                    status = "✅ PASS" if actual_success else "❌ FAIL"

                                    self.stdout.write(
                                        f"      • {metric}: {score:.3f} {status}"
                                    )

                                    # Show reason for failed tests
                                    if (
                                        not actual_success
                                        and reason
                                        and len(reason.strip()) > 0
                                    ):
                                        self.stdout.write(f"        └─ {reason}")
                self.stdout.write("    No test cases defined for this bot")

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
            metric_scores = model_avg[metric].sort_values(ascending=False)
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

    def detailed_results(
        self, df: pd.DataFrame, models: list[str], bot_names: list[str]
    ) -> None:
        """Display detailed results for each bot, model, and prompt."""
        self.stdout.write("\n\n📋 DETAILED RESULTS")
        self.stdout.write("-" * 50)

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
