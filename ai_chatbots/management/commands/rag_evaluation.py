"""Run RAG evaluations using the new evaluation framework."""

from asgiref.sync import async_to_sync
from django.core.management import BaseCommand

from ai_chatbots.evaluation.orchestrator import EvaluationOrchestrator
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
            "--no-prompts",
            dest="no_prompts",
            action="store_true",
            help="Skip using alternative prompts, only use default prompts",
        )

    def handle(self, *args, **options):  # noqa: ARG002
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
        use_prompts = not options["no_prompts"]

        # Create evaluation orchestrator
        orchestrator = EvaluationOrchestrator(self.stdout)

        # Create evaluation configuration
        config = orchestrator.create_evaluation_config(
            models=models,
            evaluation_model=evaluation_model,
        )

        # Validate bot names if provided
        if bot_names:
            bot_names = orchestrator.validate_bot_names(bot_names)
            if not bot_names:
                self.stdout.write("No valid bot names provided. Exiting.")
                return

        # Run evaluation
        async_to_sync(orchestrator.run_evaluation)(config, bot_names, use_prompts)
