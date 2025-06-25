"""Run RAG evaluations."""

import json
import os

import deepeval
import pandas as pd
from asgiref.sync import async_to_sync
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from django.core.management import BaseCommand
from langchain_core.messages import AIMessage

from ai_chatbots import chatbots
from main.test_utils import load_json_with_settings

metric_thresholds = {
    "ContextualRelevancy": 0.5,
    "ContextualRecall": 0.7,
    "Hallucination": 0.0,
    "AnswerRelevancy": 0.7,
}

bot_name_classes = {
    "recommendation": chatbots.ResourceRecommendationBot,
    "syllabus": chatbots.SyllabusBot,
    "video_gpt": chatbots.VideoGPTBot,
    "tutor": chatbots.TutorBot,
}

test_cases_per_bot = load_json_with_settings("test_json/rag_evaluation.json")


async def collect_tutor_response(
    chatbot: chatbots.BaseChatbot, case: LLMTestCase
) -> dict:
    """Collect response from TutorBot asynchronously."""
    chunks = []
    async for chunk in chatbot.get_completion(case.get("question")):
        chunks.append(chunk)  # noqa: PERF401
    return {"messages": [AIMessage(content="".join(chunks))]}


class Command(BaseCommand):
    """Run RAG evaluations."""

    help = "Run RAG evaluations"

    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            "--models",
            dest="models",
            required=False,
            help="Specify the models to test",
            default="openai/gpt-4o-mini,openai/gpt-4o",
        )
        parser.add_argument(
            "--eval_model",
            dest="eval_model",
            required=False,
            help="Specify the eval_model to test with",
            default="gpt-4o-mini",
        )
        parser.add_argument(
            "--bots",
            dest="bots",
            required=False,
            help="Specify the bots to test",
            default="",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        """Run the command"""
        evaluation_model = options["eval_model"]
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
        ]

        confident_api_key = os.environ.get("CONFIDENT_AI_API_KEY", "")
        if confident_api_key:
            deepeval.login_with_confident_api_key(confident_api_key)

        test_cases = []
        if not options["bots"]:
            bot_names = test_cases_per_bot.keys()
        else:
            bot_names = options["bots"].split(",")
        models = options["models"].split(",")
        self.stdout.write(f"Bots: {', '.join(bot_names)}, models: {', '.join(models)}")
        for bot_name in bot_names:
            for case in test_cases_per_bot[bot_name]:
                for m in models:
                    chatbot = bot_name_classes[bot_name](
                        "eval", checkpointer=None, model=m, **case.get("extra_init", {})
                    )
                    if bot_name == "tutor":
                        response = async_to_sync(collect_tutor_response)(chatbot, case)
                    else:
                        response = chatbot.agent.invoke(
                            {
                                "messages": [
                                    {"content": case.get("question"), "role": "user"}
                                ],
                                **case.get("extra_state", {}),
                            },
                            config=chatbot.config["configurable"],
                        )
                    self.stdout.write(
                        f"Response for question: {case['question']} is "
                        "{response['messages'][-1].content}"
                    )
                    tool_results = (
                        json.loads(response["messages"][2].content).get("results")
                        if case.get("expected_tools")
                        else []
                    )
                    retrieval_context = (
                        [
                            "\n\n".join(
                                f.get("chunk_content", json.dumps(f))
                                for f in tool_results
                            )
                        ]
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
                            "bot_name": bot_name,
                            "model": m,
                            **case,
                        },
                        input=case["question"],
                        actual_output=response["messages"][-1].content,
                        expected_output=case.get("expected_output", None),
                        retrieval_context=retrieval_context,
                        context=retrieval_context,
                        tools_called=tool_calls,
                        expected_tools=[
                            ToolCall(name=t) for t in case.get("expected_tools", [])
                        ],
                    )
                    test_cases.append(test_case)

        results = deepeval.evaluate(test_cases=test_cases, metrics=metrics)

        # Generate readable report
        self._generate_report(results, models, bot_names)

    def summarize_per_bot_model(self, df, models, bot_names):
        """Summarize results per bot and model."""
        self.stdout.write("\n📊 SUMMARY BY BOT AND MODEL")
        self.stdout.write("-" * 50)

        summary = (
            df.groupby(["bot", "model", "metric"])["score"].mean().unstack(fill_value=0)
        )
        for bot in bot_names:
            if bot in df["bot"].values:
                self.stdout.write(f"\n🤖 {bot.upper()} BOT:")
                bot_data = summary.loc[bot] if bot in summary.index else pd.DataFrame()

                if not bot_data.empty:
                    # Format scores with colors
                    for model in models:
                        if model in bot_data.index:
                            # Count passes from the detailed results for this
                            # bot-model combination
                            model_bot_results = df[
                                (df["bot"] == bot) & (df["model"] == model)
                            ]
                            passes = model_bot_results["success"].sum()
                            total = len(model_bot_results)

                            self.stdout.write(
                                f"\n  📱 Model: {model} "
                                f"({passes}/{total} tests passed)"
                            )
                            for metric in bot_data.columns:
                                score = bot_data.loc[model, metric]
                                # Get the actual success status from the detailed data
                                metric_result = model_bot_results[
                                    model_bot_results["metric"] == metric
                                ]
                                if not metric_result.empty:
                                    actual_success = metric_result.iloc[0]["success"]
                                    reason = metric_result.iloc[0]["reason"]
                                    status = "✅ PASS" if actual_success else "❌ FAIL"

                                self.stdout.write(
                                    f"    • {metric}: {score:.3f} {status}"
                                )
                                # Show reason for failed tests
                                if (
                                    not actual_success
                                    and reason
                                    and len(reason.strip()) > 0
                                ):
                                    # Truncate long reasons but show them
                                    display_reason = reason
                                    self.stdout.write(f"       └─ {display_reason}")
                else:
                    self.stdout.write("    No test cases defined for this bot")

    def model_comparison(self, df):
        """Compare models based on their average scores."""
        self.stdout.write("\n\n🔄 MODEL COMPARISON")
        self.stdout.write("-" * 50)

        model_avg = (
            df.groupby(["model", "metric"])["score"].mean().unstack(fill_value=0)
        )

        for metric in model_avg.columns:
            self.stdout.write(f"\n📈 {metric}:")
            metric_scores = model_avg[metric].sort_values(ascending=False)
            for i, (model, score) in enumerate(metric_scores.items()):
                self.stdout.write(f"  {i+1}. {model}: {score:.3f}")

    def overall_performance(self, df):
        """Calculate and display overall performance of each model."""
        self.stdout.write("\n\n🏆 OVERALL PERFORMANCE")
        self.stdout.write("-" * 50)

        overall_avg = df.groupby("model")["score"].mean().sort_values(ascending=False)

        for i, (model, avg_score) in enumerate(overall_avg.items()):
            self.stdout.write(f"  {i+1}. {model}: {avg_score:.3f}")

    def detailed_results(self, df, models, bot_names):
        """Display detailed results for each bot and model."""
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

                        # Group by question to show results for each test case
                        for question in model_results["question"].unique():
                            question_results = model_results[
                                model_results["question"] == question
                            ]

                            self.stdout.write(f"\n    ❓ Question: {question}")

                            for _, row in question_results.iterrows():
                                # Use the actual success value from deepeval
                                status = "✅" if row["success"] else "❌"
                                self.stdout.write(
                                    f"      {status} {row['metric']}: "
                                    f"{row['score']:.3f}"
                                )
                                if (
                                    not row["success"]
                                    and row["reason"]
                                    and len(str(row["reason"]).strip()) > 0
                                ):
                                    # Show failure reason
                                    self.stdout.write(f"         └─ {row["reason"]!s}")

    def _generate_report(self, results, models, bot_names):
        """Generate a readable evaluation report."""
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("RAG EVALUATION REPORT")
        self.stdout.write("=" * 80)

        # Create DataFrame for easier analysis
        data = [
            {
                "bot": test_result.additional_metadata["bot_name"],
                "model": test_result.additional_metadata["model"],
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

        df = pd.DataFrame(data)

        # Summary by Bot and Model
        self.summarize_per_bot_model(df, models, bot_names)

        # Model Comparison
        self.model_comparison(df)

        # Overall Performance
        self.overall_performance(df)

        # Detailed Results
        self.detailed_results(df, models, bot_names)

        self.stdout.write("\n" + "=" * 80)


# async def get_tutor_output(chatbot, case):
#     async for chunk in chatbot.get_completion(case.get("question")):
#         yield chunk
