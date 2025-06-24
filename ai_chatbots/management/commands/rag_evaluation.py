"""Run RAG evaluations."""

import json

import pandas as pd
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from django.core.management import BaseCommand

from ai_chatbots import chatbots

bot_name_classes = {
    "recommendation": chatbots.ResourceRecommendationBot,
    "syllabus": chatbots.SyllabusBot,
    "video_gpt": chatbots.VideoGPTBot,
    "tutor": chatbots.TutorBot,
}

test_cases_per_bot = {
    "recommendation": [],
    "syllabus": [
        {
            "question": "Who are the instructors for this course?",
            "extra_state": {"course_id": ["8.01SC+fall_2016"]},
            "expected_output": "The intructors for this course are "
            "Prof. Deepto Chakrabarty, Dr. Peter Dourmashkin, "
            "Dr. Michelle Tomasik, Prof. Anna Frebel, "
            "Prof. Vladan Vuletic",
            "expected_tools": [ToolCall(name="search_content_files")],
        }
    ],
    "video_gpt": [],
    "tutor": [],
}


class Command(BaseCommand):
    """Run RAG evaluations."""

    help = "Run RAG evaluations"

    def add_arguments(self, parser):
        parser.add_argument(
            "--models",
            dest="content",
            required=False,
            help="Specify the models to test",
        )
        parser.add_argument(
            "--eval_model",
            dest="eval_model",
            required=False,
            help="Specify the eval_model to test with",
            default="gpt-4o",
        )
        parser.add_argument(
            "--bots",
            dest="bots",
            required=False,
            help="Specify the bots to test",
            default="all",
        )

    def create_clean_summary(self, results):
        """Create a clean summary by properly deduplicating and organizing data"""

        # First, let's collect all unique test cases and their metrics
        test_cases = {}  # key: (model, question), value: all metrics for combination

        for test_result in results.test_results:
            model = test_result.additional_metadata.get("model", "Unknown Model")
            question = test_result.additional_metadata.get("question", test_result.name)

            key = (model, question)

            if key not in test_cases:
                test_cases[key] = {}

            # Add all metrics for this test case
            for metric in test_result.metrics_data:
                metric_name = metric.name
                # Only add if we don't already have this metric (first one wins)
                if metric_name not in test_cases[key]:
                    threshold = getattr(metric, "threshold", None)
                    passed = (
                        metric.score >= threshold if threshold is not None else None
                    )
                    test_cases[key][metric_name] = {
                        "score": metric.score,
                        "threshold": threshold,
                        "passed": passed,
                        "reason": getattr(metric, "reason", None),
                    }

        # Convert to the expected format
        summary_data = {}
        for (model, question), metrics in test_cases.items():
            if model not in summary_data:
                summary_data[model] = {}
            summary_data[model][question] = metrics

        return summary_data

    def print_clean_summary(self, summary_data):
        """Print a clean, formatted summary"""
        self.stdout.write("\n" + "=" * 120)
        self.stdout.write("CLEAN EVALUATION SUMMARY REPORT")
        self.stdout.write("=" * 120)

        if not summary_data:
            self.stdout.write("No evaluation data found.")
            return

        for model, questions in summary_data.items():
            self.stdout.write(f"\n🤖 MODEL: {model}")
            self.stdout.write("-" * 80)

            for question, metrics in questions.items():
                self.stdout.write(f"\n📝 QUESTION: {question}")
                self.stdout.write("   " + "─" * 90)

                # Create a table-like format
                self.stdout.write(
                    f"   {'Metric':<30} {'Score':<8} {'Threshold':<10} {'Status'}"
                )
                self.stdout.write("   " + "─" * 90)

                # Sort metrics for consistent display
                for metric_name in sorted(metrics.keys()):
                    data = metrics[metric_name]
                    score = data["score"]
                    threshold = data["threshold"]
                    passed = data["passed"]
                    reason = data.get("reason", "")

                    # Format score and threshold
                    score_str = f"{score:.3f}" if score is not None else "N/A"
                    threshold_str = (
                        f"{threshold:.3f}" if threshold is not None else "N/A"
                    )

                    # Determine pass/fail status
                    if passed is True:
                        status = "✅ PASS"
                    elif passed is False:
                        status = "❌ FAIL"
                    else:
                        status = "⚪ N/A"

                    self.stdout.write(
                        f"   {metric_name:<30} {score_str:<8} {threshold_str:<10} {status}"  # noqa: E501
                    )

                    # Show failure reason if test failed and reason exists
                    if passed is False and reason:
                        # Wrap long reasons to fit nicely
                        import textwrap

                        wrapped_reason = textwrap.fill(
                            reason,
                            width=80,
                            initial_indent="      💡 Reason: ",
                            subsequent_indent="           ",
                        )
                        self.stdout.write(wrapped_reason)

                self.stdout.write()

    # Create DataFrame for analysis
    def create_clean_dataframe(self, summary_data):
        """Create a pandas DataFrame from clean summary data"""
        rows = []

        for model, questions in summary_data.items():
            for question, metrics in questions.items():
                for metric_name, data in metrics.items():
                    rows.append(
                        {
                            "Model": model,
                            "Question": question,
                            "Metric": metric_name,
                            "Score": data["score"],
                            "Threshold": data["threshold"],
                            "Passed": data["passed"],
                            "Reason": data.get("reason", ""),
                        }
                    )

        return pd.DataFrame(rows)

    def format_results(self, results):
        # Generate the clean summary
        clean_summary_data = self.create_clean_summary(results)
        self.print_clean_summary(clean_summary_data)
        df_clean = self.create_clean_dataframe(clean_summary_data)

        if not df_clean.empty:
            self.stdout.write("\n" + "=" * 120)
            self.stdout.write("SUMMARY STATISTICS")
            self.stdout.write("=" * 120)

            # Show pass rates by model
            pass_rates = (
                df_clean.groupby("Model")["Passed"]
                .agg(["count", "sum", "mean"])
                .round(3)
            )
            pass_rates.columns = ["Total_Tests", "Tests_Passed", "Pass_Rate"]
            self.stdout.write("\n📊 PASS RATES BY MODEL:")
            self.stdout.write(pass_rates)

            # Show average scores by metric across all models
            avg_scores = (
                df_clean.groupby("Metric")["Score"]
                .agg(["mean", "std", "min", "max"])
                .round(3)
            )
            self.stdout.write("\n📈 AVERAGE SCORES BY METRIC:")
            self.stdout.write(avg_scores)

            # Show detailed DataFrame
            self.stdout.write("\n📋 DETAILED RESULTS:")
            self.stdout.write(df_clean.to_string(index=False))

            # Show failure analysis if there are any failures
            failures = df_clean[not df_clean["Passed"]]
            if not failures.empty:
                self.stdout.write("\n" + "=" * 120)
                self.stdout.write("FAILURE ANALYSIS")
                self.stdout.write("=" * 120)

                for _, row in failures.iterrows():
                    self.stdout.write("\n❌ FAILED TEST:")
                    self.stdout.write(f"   Model: {row['Model']}")
                    self.stdout.write(f"   Question: {row['Question']}")
                    self.stdout.write(f"   Metric: {row['Metric']}")
                    self.stdout.write(
                        f"   Score: {row['Score']:.3f} (Threshold: {row['Threshold']:.3f})"  # noqa: E501
                    )
                    if row["Reason"]:
                        import textwrap

                        wrapped_reason = textwrap.fill(
                            row["Reason"],
                            width=100,
                            initial_indent="   Reason: ",
                            subsequent_indent="           ",
                        )
                        self.stdout.write(wrapped_reason)
                    self.stdout.write()
        else:
            self.stdout.write("\nNo data available for summary statistics.")

    def handle(self, *args, **options):  # noqa: ARG002
        """Run the command"""
        evaluation_model = options["eval_model"]
        metrics = [
            ContextualRelevancyMetric(
                threshold=0.5, model=evaluation_model, include_reason=True
            ),
            ContextualRecallMetric(
                threshold=0.7, model=evaluation_model, include_reason=True
            ),
            FaithfulnessMetric(
                threshold=0.7,
                model=evaluation_model,  # 4o-mini falsely fails this test
                include_reason=True,
            ),
            HallucinationMetric(
                threshold=0.0, model=evaluation_model, include_reason=True
            ),
            AnswerRelevancyMetric(
                threshold=0.7, model=evaluation_model, include_reason=True
            ),
        ]
        test_cases = []
        if options["bots"] == "all":
            bot_names = test_cases_per_bot.keys()
        else:
            bot_names = options["bots"].split(",")
        for bot_name in bot_names:
            for case in test_cases_per_bot[bot_name]:
                for m in ["openai/gpt-4o-mini", "openai/gpt-4o"]:
                    chatbot = bot_name_classes[bot_name](
                        "eval", checkpointer=None, model=m
                    )
                    response = chatbot.agent.invoke(
                        {
                            "messages": [
                                {"content": case.get("question"), "role": "user"}
                            ],
                            **case.get("extra_state", {}),
                        },
                        config=chatbot.config["configurable"],
                    )
                    try:
                        tool_results = (
                            json.loads(response["messages"][2].content).get("results")
                            if case.get("expected_tools")
                            else []
                        )
                    except json.JSONDecodeError:
                        tool_results = []
                    retrieval_context = (
                        ["\n".join(f["chunk_content"] for f in tool_results)]
                        if tool_results
                        else []
                    )
                    tool_message = response["messages"][1] if tool_results else None
                    tool_calls = (
                        [
                            ToolCall(
                                input_parameters=json.loads(t.function.arguments),
                                name=t.function.name,
                            )
                            for t in tool_message.additional_kwargs["tool_calls"]
                        ]
                        if tool_message
                        else []
                    )
                    test_case = LLMTestCase(
                        name=f"{bot_name}-{m}",
                        additional_metadata={
                            "model": m,
                            **case,
                        },
                        input=case["question"],
                        actual_output=response["messages"][-1].content,
                    )
                    if case.get("expected_output"):
                        test_case.expected_output = case["expected_output"]
                    test_case.retrieval_context = retrieval_context
                    test_case.context = retrieval_context
                    test_case.tools_called = (tool_calls,)
                    if case.get("expected_tool_calls"):
                        test_case.expected_tools = (
                            [
                                ToolCall(
                                    input_parameters={"q": "instructor"},
                                    name="search_content_files",
                                )
                            ],
                        )  # This is the expected tool call
                    test_cases.append(test_case)

        results = evaluate(test_cases=test_cases, metrics=metrics)
        self.format_results(results)
