"""Orchestrator for running RAG evaluations across multiple bots and models."""

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path

import deepeval
from deepeval.evaluate import AsyncConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArenaGEval,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import ArenaTestCase, Contestant, LLMTestCaseParams
from django.core.management.base import OutputWrapper

from ai_chatbots.api import get_langsmith_prompt
from ai_chatbots.evaluation.base import (
    SAMPLE_SEED,
    EvaluationConfig,
    RunFailures,
    judge_error,
    sample_cases,
)
from ai_chatbots.evaluation.checks import HTML_COMMENT, run_checks
from ai_chatbots.evaluation.compare import measure_outcomes, print_comparison
from ai_chatbots.evaluation.evaluators import BOT_EVALUATORS
from ai_chatbots.evaluation.reporting import EvaluationReporter
from ai_chatbots.evaluation.timeout_wrapper import (
    TimeoutMetricWrapper,
    wrap_metrics_with_timeout,
)
from main.test_utils import load_json_with_settings

# Constants
ERROR_LOG_MIN_SIZE = 500  # Minimum file size to indicate errors beyond header

# Answers with fewer than two models' answers can't be judged pairwise
MIN_CONTESTANTS = 2


def _model_name(model_id: str) -> str:
    """Strip the provider prefix from a model id: openai/gpt-5.4 -> gpt-5.4."""
    return model_id.rsplit("/", maxsplit=1)[-1].strip().lower()


def check_judge_not_under_test(judge: str, models: list[str]) -> None:
    """Refuse a judge that is also one of the models being scored."""
    if any(_model_name(judge) == _model_name(model) for model in models):
        msg = (
            f"The judge ({judge}) is one of the models under test; "
            "choose another --eval_model."
        )
        raise ValueError(msg)


def _answer_key(metadata: dict | None) -> tuple:
    """Identify one answer: case, model and prompt."""
    metadata = metadata or {}
    return (
        metadata.get("case_id"),
        metadata.get("model"),
        metadata.get("prompt_label"),
    )


def _side(metadata: dict | None) -> str:
    """Name a baseline run's side: the model, plus the prompt unless default."""
    _, model, label = _answer_key(metadata)
    return model if label in (None, "default") else f"{model} [{label}]"


def find_prompt(prompts_data: dict, bot_name: str, name: str | None) -> dict | None:
    """Find a bot's prompt entry by name; None for the bot's own prompt."""
    if name is None:
        return None
    for prompt in prompts_data.get(bot_name, []):
        if prompt.get("name") == name:
            if bot_name == "tutor" and "{problem_statement}" not in (
                prompt.get("text") or ""
            ):
                msg = f"tutor prompt '{name}' needs text with {{problem_statement}}"
                raise ValueError(msg)
            return prompt
    msg = f"No prompt named '{name}' for {bot_name} in the prompts file"
    raise ValueError(msg)


TUTORING_QUALITY_STEPS = [
    "The input is the student's latest message; the retrieval context holds the "
    "problem, its solution if available, and the conversation so far.",
    "If the actual output gives away a final answer the student has not yet reached, "
    "or does the remaining work for them, the score must be low no matter how "
    "accurate it is, even when the student asks for the answer directly: a tutor must "
    "not hand it over.",
    "Check that every hint, explanation and claim is correct for this problem; "
    "penalize any error heavily.",
    "Check that it responds to what the student actually said, including correcting a "
    "misconception they stated or confirming an answer they got right.",
    "Check that it moves the student forward with a concrete next step, hint or "
    "guiding question, rather than generic encouragement.",
    "Check that it is clear and appropriately short for a chat.",
]


TUTOR_PAIRWISE_CRITERIA = (
    "The input is a student's latest message to a tutor; the retrieval "
    "context holds the problem, its solution if available, and the "
    "conversation so far.  Choose the tutor response that best helps the "
    "student learn: correct for this problem, responsive to what the "
    "student said, moving them forward with a concrete hint or guiding "
    "question, and not giving away the final answer or doing the "
    "remaining work for them.  Prefer clear, chat-length responses."
)


class EvaluationOrchestrator:
    """Orchestrates evaluation across multiple bots and models."""

    def __init__(self, stdout: OutputWrapper):
        self.stdout = stdout
        self.reporter = EvaluationReporter(stdout)
        self.pairwise_wins: dict[str, dict[str, int]] = {}
        # Everything the run failed to measure
        self.failures = RunFailures()
        # One record per answer: its code checks and tool calls
        self.answer_records: list[dict] = []
        # (bot, case_id, prompt_label) -> winning model, None for a judge error
        self.pairwise_results: dict[tuple, str | None] = {}
        # Per bot: how the cases were sampled, for the report
        self.sample_info: dict[str, str] = {}
        # Bots the run was asked for by name; one with no cases is a failure
        self.named_bots: set[str] = set()

    def create_evaluation_config(  # noqa: PLR0913
        self,
        models: list[str],
        evaluation_model: str,
        metric_thresholds: dict[str, float] | None = None,
        timeout_seconds: int = 360,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        *,
        require_expected: bool = False,
        correctness_only: bool = False,
        pairwise_only: bool = False,
        tutor_rubric: bool = False,
    ) -> EvaluationConfig:
        """Create evaluation configuration with metrics.

        Args:
            models: List of model IDs to evaluate.
            evaluation_model: Model to use as the evaluation judge.
            metric_thresholds: Optional custom thresholds per metric name.
            timeout_seconds: Timeout for individual metric execution.
            max_retries: Maximum retries per metric on timeout/error.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            require_expected: If True, use metrics that require curated expected
                answers (ContextualPrecision, ContextualRecall). If False
                (default), use reference-free metrics (Faithfulness, GEval)
                that evaluate quality without expected answers.
            correctness_only: If True, judge the answer against the curated
                expected answer and nothing else.  The other metrics score the
                answer against the retrieval context, which makes them
                unusable for comparing two versions of the code that reach the
                same answer through different tools, or through none at all.
        """
        check_judge_not_under_test(evaluation_model, models)
        if pairwise_only:
            metric_thresholds = metric_thresholds or {}
            metrics = []
        elif correctness_only or tutor_rubric:
            name = "TutoringQuality" if tutor_rubric else "Correctness"
            if metric_thresholds is None:
                metric_thresholds = {name: 0.7}
            metrics = [
                GEval(
                    name=name,
                    # Fixed steps, since GEval writes its own from criteria on
                    # every run, and expected outputs are written as rubrics
                    evaluation_steps=TUTORING_QUALITY_STEPS
                    if tutor_rubric
                    else [
                        "Read the expected output as a rubric for the answer, "
                        "not as text the actual output must repeat. Wording, "
                        "order and format don't matter.",
                        "Check that every fact the expected output requires is "
                        "in the actual output: numbers, grades, prices, dates, "
                        "names and links must match.",
                        "Find each thing the expected output says the answer "
                        "doesn't or shouldn't do (such as 'It doesn't list "
                        "other courses'), and check whether the actual output "
                        "does it anyway. The actual output need not say it "
                        "avoids it. Anything the expected output says the "
                        "answer may do is optional.",
                        "Extra detail that doesn't contradict the expected "
                        "output is acceptable and must not lower the score.",
                        "Score low if the actual output leaves out or gets "
                        "wrong a required fact, does something the expected "
                        "output says it doesn't, answers about a different "
                        "course, program or platform than the one asked "
                        "about, or says no information could be found when "
                        "the expected output supplies it.",
                    ],
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        # The tutor is judged against the problem and conversation
                        LLMTestCaseParams.RETRIEVAL_CONTEXT
                        if tutor_rubric
                        else LLMTestCaseParams.EXPECTED_OUTPUT,
                    ],
                    model=evaluation_model,
                    threshold=metric_thresholds[name],
                ),
            ]
        elif require_expected:
            if metric_thresholds is None:
                metric_thresholds = {
                    "ContextualPrecision": 0.7,
                    "ContextualRelevancy": 0.5,
                    "ContextualRecall": 0.7,
                    "Hallucination": 0.8,
                    "AnswerRelevancy": 0.7,
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
        else:
            if metric_thresholds is None:
                metric_thresholds = {
                    "Faithfulness": 0.7,
                    "ContextualRelevancy": 0.5,
                    "Hallucination": 0.8,
                    "AnswerRelevancy": 0.7,
                    "Helpfulness": 0.7,
                }

            metrics = [
                FaithfulnessMetric(
                    threshold=metric_thresholds["Faithfulness"],
                    model=evaluation_model,
                    include_reason=True,
                ),
                ContextualRelevancyMetric(
                    threshold=metric_thresholds["ContextualRelevancy"],
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
                GEval(
                    name="Helpfulness",
                    criteria=(
                        "Determine whether the response is helpful, accurate, "
                        "and appropriate for an educational context. Consider "
                        "whether it directly addresses the user's question, "
                        "provides clear and understandable information, and "
                        "would be useful to a student or learner."
                    ),
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    model=evaluation_model,
                    threshold=metric_thresholds["Helpfulness"],
                ),
            ]

        # Wrap metrics with timeout and retry functionality
        timeout_wrapped_metrics = wrap_metrics_with_timeout(
            metrics, timeout_seconds, max_retries, retry_delay
        )

        return EvaluationConfig(
            models=models,
            evaluation_model=evaluation_model,
            metrics=timeout_wrapped_metrics,
            metric_thresholds=metric_thresholds,
            confident_api_key=os.environ.get("CONFIDENT_AI_API_KEY"),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay=retry_delay,
            correctness_only=correctness_only and not pairwise_only,
            pairwise_only=pairwise_only,
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
            self.failures.add(
                "all bots", "prompts file not loaded", detail=prompts_file
            )
        return prompts_data

    def _load_bot_cases(  # noqa: PLR0913
        self,
        evaluator,
        bot_name: str,
        config: EvaluationConfig,
        sample: int | None,
        seed: int,
        data_file: str | list[str] | None = None,
    ) -> list:
        """Load a bot's cases from every data file and take the run's sample."""
        cases = evaluator.load_cases(require_reference=config.correctness_only)
        picked = sample_cases(cases, sample, seed)
        info = f"{len(picked)} of {len(cases)} cases"
        if len(picked) < len(cases):
            info += f", sampled with seed {seed}"
        self.sample_info[bot_name] = info
        self.stdout.write(f"Loaded {info} for {bot_name}")
        if not picked and bot_name in self.named_bots:
            files = (
                data_file if isinstance(data_file, str) else ",".join(data_file or [])
            )
            self.failures.add(bot_name, "no cases loaded", detail=files or "default")
        return picked

    def _record_answers(self, bot_name: str, test_cases: list) -> None:
        """Note each answer's code checks and anything that went wrong."""
        add = self.failures.add
        for case in test_cases:
            metadata = case.metadata or {}
            # Ignore the metadata comments some bots append
            if not HTML_COMMENT.sub("", case.actual_output or "").strip():
                add(bot_name, "empty answer", case)
            for tool in metadata.get("search_errors") or []:
                add(bot_name, "search tool returned an error", case, detail=tool)
            for call in metadata.get("blocked_account_calls") or []:
                add(
                    bot_name,
                    "account tool called without made-up records",
                    case,
                    detail=call,
                )
            case_id, model, prompt_label = _answer_key(metadata)
            self.answer_records.append(
                {
                    "bot": bot_name,
                    "question": case.input,
                    "case_id": case_id,
                    "model": model,
                    "prompt_label": prompt_label,
                    "side": _side(metadata),
                    "checks": run_checks(bot_name, case),
                    "tool_call_count": metadata.get("tool_call_count"),
                }
            )

    async def _collect_and_evaluate_bot(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        bot_name: str,
        config: EvaluationConfig,
        *,
        data_file: str | list[str] | None,
        use_prompts: bool | None,
        prompts_data: dict | None,
        max_concurrent: int,
        batch_size: int,
        error_log_file: str,
        pairwise: bool = False,
        sample: int | None = None,
        seed: int = SAMPLE_SEED,
        sides: list[tuple[str, dict | None]] | None = None,
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

        # Load, filter and sample the test cases for this bot
        bot_test_cases = self._load_bot_cases(
            evaluator, bot_name, config, sample, seed, data_file
        )

        # Get prompts for this bot (default + alternatives)
        bot_prompts = [None]  # Default prompt (None means use bot's default)
        if use_prompts and bot_name in prompts_data:
            bot_prompts.extend(prompts_data[bot_name])

        all_llm_test_cases = []
        pairwise_cases = []
        batch_results = []

        # Evaluate each model with each prompt, or a baseline run's two sides
        runs = sides or [(m, p) for m in config.models for p in bot_prompts]
        for model, prompt in runs:
            # Resolve prompt details
            if prompt is None:
                prompt_label = "default"
                prompt_text = None
                source = "bot default"
            else:
                prompt_label = prompt["name"]
                source = "prompts file" if prompt.get("text") else "LangSmith"
                prompt_text = prompt.get("text") or get_langsmith_prompt(prompt_label)

            if prompt_label != "default" and not prompt_text:
                self.stdout.write(f"No prompt text for '{prompt_label}', skipping")
                self.failures.add(
                    bot_name,
                    "prompt skipped: no prompt text",
                    model=model,
                    prompt_label=prompt_label,
                )
                continue

            if prompt_text:
                digest = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
                source += f", sha256 {digest}"
            self.stdout.write(
                f"Evaluating {bot_name} with {model} using {prompt_label} ({source})"
            )

            try:
                # Generate bot responses concurrently (this is now parallel!)
                model_test_cases = await evaluator.evaluate_model(
                    model,
                    bot_test_cases,
                    instructions=prompt_text,
                    prompt_label=prompt_label,
                    max_concurrent=max_concurrent,
                    failures=self.failures,
                )
                self._record_answers(bot_name, model_test_cases)
                if config.metrics:
                    all_llm_test_cases.extend(model_test_cases)
                if pairwise:
                    pairwise_cases.extend(model_test_cases)

                # Process batches (loop until remaining < batch_size)
                while batch_size > 0 and len(all_llm_test_cases) >= batch_size:
                    batch = all_llm_test_cases[:batch_size]
                    remaining = all_llm_test_cases[batch_size:]

                    batch_num = len(batch_results) + 1
                    self.stdout.write(
                        f"\n===== Evaluating batch {batch_num} for {bot_name} "
                        f"({len(batch)} test cases) ====="
                    )

                    batch_result = await asyncio.to_thread(
                        self._run_deepeval_evaluation,
                        batch,
                        config,
                        max_concurrent,
                        bot_name=bot_name,
                    )
                    batch_results.append(batch_result)

                    # Clear processed test cases to free memory
                    all_llm_test_cases = remaining

                    self.stdout.write(
                        f"Batch {batch_num} complete: "
                        f"{len(batch_result.test_results)} results"
                    )

            except Exception as e:
                msg = f"Error on {bot_name} with {model} and {prompt_label}: {e}"
                raise RuntimeError(msg) from e

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

            final_result = await asyncio.to_thread(
                self._run_deepeval_evaluation,
                all_llm_test_cases,
                config,
                max_concurrent,
                bot_name=bot_name,
            )
            batch_results.append(final_result)

        if pairwise:
            await asyncio.to_thread(
                self._run_pairwise,
                bot_name,
                pairwise_cases,
                config,
                by_side=bool(sides),
            )

        # Merge all batch results for this bot
        bot_result = self._merge_evaluation_results(batch_results)
        self.stdout.write(
            f"\nCompleted evaluation for {bot_name}: "
            f"{len(bot_result.test_results)} total results"
        )

        return bot_result

    def _run_deepeval_evaluation(  # noqa: C901
        self,
        test_cases: list,
        config: EvaluationConfig,
        max_concurrent: int,
        *,
        bot_name: str | None = None,
    ) -> EvaluationResult:
        """Run the actual DeepEval evaluation, recording what it couldn't judge.

        Callers must run this off the event loop (asyncio.to_thread). Called
        from a coroutine, deepeval.evaluate() falls back to nest_asyncio and
        re-enters the running loop, which on python 3.14 fails the first API
        call and then hangs the retries indefinitely.
        """
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

        for metric in config.metrics:
            if isinstance(metric, TimeoutMetricWrapper):
                metric.prepare()
        result = deepeval.evaluate(
            test_cases=test_cases,
            metrics=config.metrics,
            error_config=ErrorConfig(ignore_errors=True),
            async_config=async_config,
        )
        if len(result.test_results) < len(test_cases):
            self.stdout.write(
                f"⚠️  deepeval returned {len(result.test_results)} of "
                f"{len(test_cases)} test cases"
            )
        returned = {
            _answer_key(tr.metadata): tr
            for tr in result.test_results
            if isinstance(getattr(tr, "metadata", None), dict)
        }
        for case in test_cases:
            bot = bot_name or (case.metadata or {}).get("bot_name", "unknown")
            test_result = returned.get(_answer_key(case.metadata))
            if test_result is None:
                self.failures.add(bot, "deepeval returned no result", case)
                continue
            metrics_data = getattr(test_result, "metrics_data", None) or []
            for metric_data in metrics_data:
                if error := judge_error(metric_data):
                    self.failures.add(
                        bot, "judge error", case, detail=f"{metric_data.name}: {error}"
                    )
            for name in {m.__name__ for m in config.metrics} - {
                m.name for m in metrics_data
            }:
                self.failures.add(bot, "judge returned no score", case, detail=name)
        return result

    def _run_pairwise(
        self,
        bot_name: str,
        test_cases: list,
        config: EvaluationConfig,
        *,
        by_side: bool = False,
    ) -> None:
        """Ask the judge which model answered each case best.

        No expected answer is involved.  deepeval shows the judge the answers in
        random order and under fake names, which reduces position and name bias.
        """
        if len(config.models) < MIN_CONTESTANTS:
            self.stdout.write("⚠️  --pairwise needs two or more models; skipping")
            self.failures.add(bot_name, "pairwise skipped: needs two or more models")
            return
        arena_metric = ArenaGEval(
            name="PairwisePreference",
            criteria=TUTOR_PAIRWISE_CRITERIA
            if bot_name == "tutor"
            else (
                "Choose the response that best serves the learner who asked the "
                "input question.  Prefer responses that are accurate and fully "
                "supported by the retrieval context (penalize any claim the "
                "context does not support), that directly address what was "
                "asked including qualifiers such as 'basic' or 'free', and that "
                "are clear and appropriately concise.  For tutoring requests, "
                "prefer responses that guide the learner toward the solution "
                "with hints or questions over ones that give the method or "
                "answer away."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=config.evaluation_model,
            async_mode=False,
        )
        metric = TimeoutMetricWrapper(
            arena_metric,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_delay,
        )
        metric.prepare()
        groups: dict[tuple, dict] = {}
        for case in test_cases:
            case_id, model, prompt_label = _answer_key(case.metadata)
            if by_side:
                groups.setdefault((case_id, None), {})[_side(case.metadata)] = case
            else:
                groups.setdefault((case_id, prompt_label), {})[model] = case
        names = (n for g in groups.values() for n in g) if by_side else config.models
        wins = dict.fromkeys(names, 0)
        self.stdout.write(f"\n===== Pairwise preference for {bot_name} =====")
        for (case_id, prompt_label), by_model in groups.items():
            some_case = next(iter(by_model.values()))
            question = some_case.input
            # Not one model's failure, so model is left out
            where = {"case": some_case, "model": None}
            if len(by_model) < MIN_CONTESTANTS:
                missing = sorted(
                    set(config.models)
                    - {c.metadata["model"] for c in by_model.values()}
                )
                self.failures.add(
                    bot_name,
                    "pairwise skipped: an answer is missing",
                    detail=", ".join(missing),
                    **where,
                )
                continue
            arena = ArenaTestCase(
                contestants=[
                    Contestant(name=model, test_case=case)
                    for model, case in by_model.items()
                ]
            )
            winner = metric.measure(arena)
            if metric.error:
                self.stdout.write(
                    f"  ❓ {question[:70]}\n     ⚠️ judge error: {metric.error}"
                )
                self.failures.add(
                    bot_name, "pairwise judge error", detail=str(metric.error), **where
                )
                self.pairwise_results[(bot_name, case_id, prompt_label)] = None
                continue
            wins[winner] = wins.get(winner, 0) + 1
            self.pairwise_results[(bot_name, case_id, prompt_label)] = winner
            self.stdout.write(
                f"  ❓ {question[:70]}\n     🏆 {winner}: {metric.reason}"
            )
            record = {
                "bot": bot_name,
                "question": question,
                "case_key": case_id,
                "winner": winner,
                "reason": metric.reason,
            }
            self.stdout.write(f"PAIRWISE_JSON {json.dumps(record)}")
        self.stdout.write(f"Pairwise wins for {bot_name}: {wins}")
        self.pairwise_wins[bot_name] = wins

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
{"#" * 80}
RAG EVALUATION ERROR LOG
Started: {timestamp}
{"#" * 80}

"""
            with log_path.open("w", encoding="utf-8") as f:
                f.write(header)

            self.stdout.write(f"Error log initialized: {error_log_file}")
        except Exception as e:  # noqa: BLE001
            self.stdout.write(f"Warning: Failed to initialize error log: {e}")

    def _versus_baseline(
        self, bot_name: str, config: EvaluationConfig, sides: list, results
    ) -> None:
        """Print how the candidate side did against the baseline side on one bot."""
        baseline, candidate = (
            _side({"model": model, "prompt_label": prompt["name"] if prompt else None})
            for model, prompt in sides
        )
        scores: dict[str, dict] = {}
        for test_result in results.test_results:
            metadata = getattr(test_result, "metadata", None) or {}
            for metric_data in test_result.metrics_data or []:
                # deepeval names it "Correctness [GEval]"
                name = metric_data.name.split(" [")[0]
                if (
                    name in config.metric_thresholds
                    and metadata.get("bot_name") == bot_name
                    and not judge_error(metric_data)
                ):
                    key = (metadata.get("case_id"), _side(metadata))
                    scores.setdefault(name, {})[key] = metric_data.score
        outcomes, questions = measure_outcomes(
            [r for r in self.answer_records if r["bot"] == bot_name],
            scores,
            config.metric_thresholds,
            {
                c: w
                for (bot, c, _), w in self.pairwise_results.items()
                if bot == bot_name
            },
            baseline,
            candidate,
            skip={case_id for case_id, _ in self.failures.failed_cases(bot_name)},
        )
        self.stdout.write(
            f"\n⚖️  {bot_name.upper()}: {candidate} versus baseline {baseline}, "
            f"on {len(questions)} cases ({self.sample_info.get(bot_name, '')})"
        )
        print_comparison(
            self.stdout.write,
            outcomes,
            questions,
            failed=bool(self.failures.for_bot(bot_name)),
        )

    async def run_evaluation(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        config: EvaluationConfig,
        *,
        bot_names: list[str] | None = None,
        data_file: str | list[str] | None = None,
        use_prompts: bool | None = True,
        prompts_file: str | None = None,
        max_concurrent: int | None = 10,
        batch_size: int | None = 0,
        error_log_file: str | None = None,
        pairwise: bool = False,
        sample: int | None = None,
        seed: int = SAMPLE_SEED,
        baseline: str | None = None,
        baseline_prompt: str | None = None,
        candidate_prompt: str | None = None,
        run_info: dict | None = None,
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
        self.named_bots = set(bot_names or [])
        bot_names = bot_names or list(BOT_EVALUATORS.keys())

        # Load alternative prompts if enabled
        prompts_data = self._load_prompts_data(prompts_file) if use_prompts else {}
        # A baseline run's (model, prompt) sides per bot, looked up before any
        # answer is generated so a bad prompt name fails fast
        sides_by_bot = {}
        if baseline:
            candidate = next(m for m in config.models if m != baseline)
            sides_by_bot = {
                bot: [
                    (baseline, find_prompt(prompts_data, bot, baseline_prompt)),
                    (candidate, find_prompt(prompts_data, bot, candidate_prompt)),
                ]
                for bot in bot_names
            }

        # Process each bot one at a time, accumulating results
        all_bot_results = []
        for bot_name in bot_names:
            self.stdout.write(f"\n{'=' * 60}")
            self.stdout.write(f"Processing bot: {bot_name}")
            self.stdout.write(f"{'=' * 60}")

            bot_result = await self._collect_and_evaluate_bot(
                bot_name,
                config,
                data_file=data_file,
                use_prompts=use_prompts,
                prompts_data=prompts_data,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                error_log_file=error_log_file,
                pairwise=pairwise or bool(baseline),
                sample=sample,
                seed=seed,
                sides=sides_by_bot.get(bot_name),
            )
            all_bot_results.append(bot_result)

        # Merge results from all bots
        results = self._merge_evaluation_results(all_bot_results)

        # Log final summary
        self.stdout.write(f"\n{'=' * 60}")
        self.stdout.write("EVALUATION COMPLETE - FINAL SUMMARY")
        self.stdout.write(f"{'=' * 60}")
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
            model = tr.metadata.get("model", "unknown")
            bot = tr.metadata.get("bot_name", "unknown")
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
        metric_names = [m.__name__ for m in config.metrics]
        self.reporter.generate_report(
            results,
            config.models,
            bot_names,
            use_prompts=use_prompts,
            metric_thresholds=config.metric_thresholds,
            evaluation_model=config.evaluation_model,
            metric_names=metric_names,
            failures=self.failures,
            answer_records=self.answer_records,
            run_info={
                **(run_info or {}),
                **{f"Cases ({bot})": info for bot, info in self.sample_info.items()},
            },
        )
        if self.pairwise_wins:
            self.stdout.write("\n🥊 PAIRWISE WINS (judge, no expected answer)")
            for bot, wins in self.pairwise_wins.items():
                self.stdout.write(f"  {bot}: {wins}")
        if baseline:
            self.stdout.write("\n\n⚖️  VERSUS BASELINE")
            self.stdout.write("=" * 80)
            for bot_name in bot_names:
                self._versus_baseline(bot_name, config, sides_by_bot[bot_name], results)
        self.reporter.incomplete_banner(self.failures)

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
