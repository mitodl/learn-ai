"""Run RAG evaluations using the new evaluation framework."""

from pathlib import Path

from asgiref.sync import async_to_sync
from django.core.management import BaseCommand, CommandError

from ai_chatbots.evaluation.base import SAMPLE_SEED
from ai_chatbots.evaluation.evaluators import BOT_EVALUATORS
from ai_chatbots.evaluation.orchestrator import EvaluationOrchestrator
from ai_chatbots.evaluation.reporting import DualOutputWrapper
from ai_chatbots.models import LLMModel


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
                "Judge answers against their curated expected answers, plus "
                "Faithfulness to the retrieved material for the syllabus and "
                "video bots.  Use this to compare two versions of the code: "
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
            help="Compare the other model in --models with this one; give a verdict",
        )
        parser.add_argument(
            "--baseline-prompt", help="A text file with the baseline's prompt"
        )
        parser.add_argument(
            "--candidate-prompt", help="A text file with the candidate's prompt"
        )
        parser.add_argument(
            "--sample",
            type=int,
            help="Judge a seeded sample of this many cases per bot (default: all)",
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
        if side_prompts and not (
            options["baseline"] and options["bots"] in BOT_EVALUATORS
        ):
            msg = "--baseline-prompt and --candidate-prompt need --baseline and one bot"
            raise CommandError(msg)
        names = [Path(f).name for f in (options["data_file"] or "").split(",") if f]
        if len(names) != len(set(names)):
            msg = "--data-file names must differ: eval cases are named by file"
            raise CommandError(msg)
        if options["pairwise_only"] and len(models) < 2:  # noqa: PLR2004
            msg = "--pairwise-only needs two or more models"
            raise CommandError(msg)
        baseline = options["baseline"]
        if baseline:
            if len(models) != 2:  # noqa: PLR2004
                msg = "--baseline needs exactly one other model in --models"
                raise CommandError(msg)
            if options["prompts"] or options["prompts_file"]:
                msg = (
                    "--baseline runs one prompt per side; use --baseline-prompt "
                    "or --candidate-prompt instead of --prompts"
                )
                raise CommandError(msg)

    def _tutor_rubric(self, options: dict) -> bool:
        """Tutor alone, with no other metric mode, is judged on TutoringQuality."""
        modes = ("pairwise_only", "correctness_only", "require_expected")
        return options["bots"] == "tutor" and not any(options[m] for m in modes)

    def _side_prompt(self, path: str | None, bot: str) -> dict | None:
        """Read a side's prompt file; the file name labels it."""
        if path is None:
            return None
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            msg = f"{path} is empty"
            raise CommandError(msg)
        if bot == "tutor" and "{problem_statement}" not in text:
            msg = f"{path}: a tutor prompt needs {{problem_statement}}"
            raise CommandError(msg)
        return {"name": Path(path).stem, "text": text}

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
        bots = options["bots"]
        bot_names = bots.split(",") if bots else None
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
        if baseline:
            models = list(dict.fromkeys([baseline, *models]))
        self._check_options(options, models)
        # A comparison needs no metric flag: pairwise, or TutoringQuality for tutor
        tutor_rubric = self._tutor_rubric(options)
        pairwise_only = options["pairwise_only"] or bool(
            baseline and not (correctness_only or require_expected or tutor_rubric)
        )
        sample = options["sample"]

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
                    pairwise_only=pairwise_only,
                    tutor_rubric=tutor_rubric,
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
                pairwise=options["pairwise"] or pairwise_only,
                sample=sample,
                seed=options["seed"],
                baseline=baseline,
                baseline_prompt=self._side_prompt(options["baseline_prompt"], bots),
                candidate_prompt=self._side_prompt(options["candidate_prompt"], bots),
            )
        finally:
            # Clean up file resources if using DualOutputWrapper
            if output_file and hasattr(output_wrapper, "close"):
                output_wrapper.close()
