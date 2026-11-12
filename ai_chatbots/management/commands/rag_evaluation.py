"""Run RAG evaluations using the new evaluation framework."""

from asgiref.sync import async_to_sync
from django.core.management import BaseCommand, CommandError

from ai_chatbots.evaluation.base import SAMPLE_SEED
from ai_chatbots.evaluation.orchestrator import EvaluationOrchestrator
from ai_chatbots.evaluation.reporting import DualOutputWrapper
from ai_chatbots.models import LLMModel

# Cases judged per run when comparing with a baseline, unless --sample says
BASELINE_SAMPLE = 30


class Command(BaseCommand):
    """Run RAG evaluations."""

    help = "Run RAG evaluations using the new evaluation framework"

    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            "--models",
            dest="models",
            required=False,
            help="Specify the models to test",
            default="",
        )
        parser.add_argument(
            "--eval_model",
            dest="eval_model",
            required=False,
            help=(
                "The judge model (default: gpt-5.5). It must not be one of "
                "the models under test."
            ),
            default="gpt-5.5",
        )
        parser.add_argument(
            "--bots",
            dest="bots",
            required=False,
            help="Specify the bots to test",
            default="",
        )
        parser.add_argument(
            "--data-file",
            dest="data_file",
            required=False,
            help="The data file to use, or several separated by commas",
            default=None,
        )
        parser.add_argument(
            "--prompts",
            dest="prompts",
            action="store_true",
            help="Include alternative prompts in addition to default prompts",
        )
        parser.add_argument(
            "--prompts-file",
            dest="prompts_file",
            required=False,
            help="Specify the prompts file to use",
            default=None,
        )
        parser.add_argument(
            "--output-file",
            dest="output_file",
            required=False,
            help="Specify a file to save the evaluation report (in addition to "
            "console output)",
            default=None,
        )
        parser.add_argument(
            "--timeout",
            dest="timeout",
            required=False,
            type=int,
            help="Timeout in seconds for individual test cases (default: 360)",
            default=360,
        )
        parser.add_argument(
            "--max-concurrent",
            dest="max_concurrent",
            required=False,
            type=int,
            help="Maximum number of test cases to run in parallel (default: 1)",
            default=1,
        )
        parser.add_argument(
            "--batch-size",
            dest="batch_size",
            required=False,
            type=int,
            help="Number of test cases to evaluate per batch to reduce memory usage "
            "(default: 10, 0 = no batching)",
            default=10,
        )
        parser.add_argument(
            "--error-log-file",
            dest="error_log_file",
            required=False,
            help=(
                "Specify a file to save error logs (default: rag_evaluation_errors.log)"
            ),
            default="rag_evaluation_errors.log",
        )
        parser.add_argument(
            "--require-expected",
            dest="require_expected",
            action="store_true",
            help=(
                "Use metrics that require curated expected answers "
                "(ContextualPrecision, ContextualRecall). Without this flag, "
                "reference-free metrics are used (Faithfulness, GEval) that "
                "evaluate quality without expected answers."
            ),
        )
        parser.add_argument(
            "--correctness-only",
            dest="correctness_only",
            action="store_true",
            help=(
                "Judge answers against their curated expected answers and "
                "nothing else.  Use this to compare two versions of the code: "
                "the other metrics score answers against the retrieval "
                "context, so they penalize an answer that needed fewer tool "
                "calls, or none."
            ),
        )
        parser.add_argument(
            "--pairwise",
            dest="pairwise",
            action="store_true",
            help=(
                "Also ask the judge which model answered each question best, "
                "without an expected answer.  Needs two or more models."
            ),
        )
        parser.add_argument(
            "--pairwise-only",
            action="store_true",
            help="Judge only pairwise, for cases without expected answers",
        )
        parser.add_argument(
            "--baseline",
            help=(
                "Compare the other model with this one and give a verdict; needs "
                "--correctness-only or --pairwise-only, except for --bots tutor"
            ),
        )
        parser.add_argument(
            "--baseline-prompt", help="The baseline's prompt, by name in --prompts-file"
        )
        parser.add_argument(
            "--candidate-prompt", help="The candidate's prompt, like --baseline-prompt"
        )
        parser.add_argument(
            "--sample",
            type=int,
            help=f"Seeded cases per bot (all; {BASELINE_SAMPLE} with --baseline)",
        )
        parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
        parser.add_argument(
            "--max-retries",
            dest="max_retries",
            required=False,
            type=int,
            help="Max retries per metric on timeout/error (default: 3)",
            default=3,
        )
        parser.add_argument(
            "--retry-delay",
            dest="retry_delay",
            required=False,
            type=float,
            help=(
                "Base delay in seconds between retries with exponential "
                "backoff and jitter (default: 5.0)"
            ),
            default=5.0,
        )

    def _check_options(self, options: dict, models: list[str]) -> None:
        """Refuse flag combinations without a meaningful result."""
        if options["pairwise_only"] and (
            options["correctness_only"] or options["require_expected"]
        ):
            msg = "--pairwise-only can't be combined with other metric modes"
            raise CommandError(msg)
        side_prompts = options["baseline_prompt"] or options["candidate_prompt"]
        if side_prompts and not (options["baseline"] and options["prompts_file"]):
            msg = (
                "--baseline-prompt and --candidate-prompt need --baseline "
                "and --prompts-file"
            )
            raise CommandError(msg)
        if options["pairwise_only"] and len(models) < 2:  # noqa: PLR2004
            msg = "--pairwise-only needs two or more models"
            raise CommandError(msg)
        baseline = options["baseline"]
        if baseline:
            if len(models) != 2 or baseline not in models:  # noqa: PLR2004
                msg = "--baseline must be one of exactly two --models"
                raise CommandError(msg)
            if not (
                options["correctness_only"]
                or options["pairwise_only"]
                or self._tutor_rubric(options)
            ):
                msg = "--baseline needs --correctness-only or --pairwise-only"
                raise CommandError(msg)
            if options["prompts"] or (options["prompts_file"] and not side_prompts):
                msg = (
                    "--baseline runs one prompt per side; use --baseline-prompt "
                    "or --candidate-prompt instead of --prompts"
                )
                raise CommandError(msg)

    def _tutor_rubric(self, options: dict) -> bool:
        """Tutor alone, with no other metric mode, is judged on TutoringQuality."""
        modes = ("pairwise_only", "correctness_only", "require_expected")
        return options["bots"] == "tutor" and not any(options[m] for m in modes)

    def _model_settings(self, models: list[str]) -> str:
        """Describe each model's temperature and reasoning effort."""
        rows = {r.litellm_id: r for r in LLMModel.objects.filter(litellm_id__in=models)}

        def describe(model):
            row = rows.get(model)
            default = "provider default"
            temp = (
                default if row is None or row.temperature is None else row.temperature
            )
            effort = (row.reasoning_effort if row else "") or default
            return f"{model} (temperature {temp}, reasoning effort {effort})"

        return "; ".join(describe(model) for model in models)

    def handle(self, **options):
        """Run the command using the new evaluation framework."""
        # Parse command line arguments
        models = [m for m in options["models"].split(",") if m] or (
            # If you don't specify models, all the enabled models will be used;
            # so make sure you have any necessary API keys set in your environment.
            list(
                LLMModel.objects.filter(enabled=True).values_list(
                    "litellm_id", flat=True
                )
            )
        )
        evaluation_model = options["eval_model"]
        bot_names = options["bots"].split(",") if options["bots"] else None
        data_file = options["data_file"] if options["data_file"] else None
        use_prompts = options["prompts"] or options["prompts_file"] is not None
        prompts_file = options["prompts_file"]
        output_file = options["output_file"]
        timeout_seconds = options["timeout"]
        max_concurrent = options["max_concurrent"]
        batch_size = options["batch_size"]
        error_log_file = options["error_log_file"]
        require_expected = options["require_expected"]
        correctness_only = options["correctness_only"]
        max_retries = options["max_retries"]
        retry_delay = options["retry_delay"]
        baseline = options["baseline"]
        self._check_options(options, models)
        sample = options["sample"]
        if sample is None and baseline:
            sample = BASELINE_SAMPLE
        run_info = {
            "Data files": data_file or "test_json/rag_evaluation.json",
            "Model settings": self._model_settings(models),
            "Sample": (
                f"up to {sample} cases per bot, seed {options['seed']}"
                if sample
                else "every case"
            ),
        }
        if baseline:
            run_info["Baseline"] = (
                f"{baseline}, prompt {options['baseline_prompt'] or 'default'}"
            )
            run_info["Candidate prompt"] = options["candidate_prompt"] or "default"

        # Create output wrapper (dual output if file specified, otherwise normal stdout)
        if output_file:
            output_wrapper = DualOutputWrapper(self.stdout, output_file)
            self.stdout.write(f"Report will be saved to: {output_file}")
        else:
            output_wrapper = self.stdout

        try:
            # Create evaluation orchestrator with the appropriate output wrapper
            orchestrator = EvaluationOrchestrator(output_wrapper)

            # Create evaluation configuration
            try:
                config = orchestrator.create_evaluation_config(
                    models=models,
                    evaluation_model=evaluation_model,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    require_expected=require_expected,
                    correctness_only=correctness_only,
                    pairwise_only=options["pairwise_only"],
                    tutor_rubric=self._tutor_rubric(options),
                )
            except ValueError as error:
                raise CommandError(str(error)) from error

            # Validate bot names if provided
            if bot_names:
                bot_names = orchestrator.validate_bot_names(bot_names)
                if not bot_names:
                    self.stdout.write("No valid bot names provided. Exiting.")
                    return

            # Run evaluation
            async_to_sync(orchestrator.run_evaluation)(
                config,
                bot_names=bot_names,
                data_file=data_file,
                use_prompts=use_prompts,
                prompts_file=prompts_file,
                max_concurrent=max_concurrent,
                batch_size=batch_size,
                error_log_file=error_log_file,
                pairwise=options["pairwise"] or options["pairwise_only"],
                sample=sample,
                seed=options["seed"],
                baseline=baseline,
                baseline_prompt=options["baseline_prompt"],
                candidate_prompt=options["candidate_prompt"],
                run_info=run_info,
            )
        finally:
            # Clean up file resources if using DualOutputWrapper
            if output_file and hasattr(output_wrapper, "close"):
                output_wrapper.close()
