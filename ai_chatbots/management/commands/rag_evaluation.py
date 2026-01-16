"""Run RAG evaluations using the new evaluation framework."""

from asgiref.sync import async_to_sync
from django.core.management import BaseCommand

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
        parser.add_argument(
            "--data-file",
            dest="data_file",
            required=False,
            help="Specify the data file to use",
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
            "(default: 0 = no batching, all at once)",
            default=10,
        )
        parser.add_argument(
            "--error-log-file",
            dest="error_log_file",
            required=False,
            help=(
                "Specify a file to save error logs "
                "(default: rag_evaluation_errors.log)"
            ),
            default="rag_evaluation_errors.log",
        )

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
            config = orchestrator.create_evaluation_config(
                models=models,
                evaluation_model=evaluation_model,
                timeout_seconds=timeout_seconds,
            )

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
            )
        finally:
            # Clean up file resources if using DualOutputWrapper
            if output_file and hasattr(output_wrapper, "close"):
                output_wrapper.close()
