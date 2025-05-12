"""Management command for clearing the prompt cache."""

from django.core.management import BaseCommand
from open_learning_ai_tutor.prompts import (
    assessment_prompt_mapping,
    intent_prompt_mapping,
    prompt_env_key,
)

from ai_chatbots.prompts import PROMPT_MAPPING
from ai_chatbots.utils import get_django_cache

all_prompt_keys = [
    prompt_env_key(key)
    for key in (
        list(intent_prompt_mapping.keys())
        + list(assessment_prompt_mapping.keys())
        + list(PROMPT_MAPPING.keys())
        + ["tutor_initial_assessment", "tutor_problem"]
    )
]


class Command(BaseCommand):
    """Clear the prompt cache."""

    help = "Clear the prompt cache."
    cache = get_django_cache()

    def add_arguments(self, parser):
        parser.add_argument(
            "--prompts",
            dest="prompt_names",
            required=False,
            help="If set, clear the cache for only these prompts",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        """Clear the prompt cache."""
        prompt_names = (
            [prompt_env_key(option) for option in options["prompt_names"].split(",")]
            if options["prompt_names"]
            else all_prompt_keys
        )

        for prompt_name in prompt_names:
            self.cache.delete(prompt_name)
            self.stdout.write(self.style.SUCCESS(f"Cleared cache for {prompt_name}"))
