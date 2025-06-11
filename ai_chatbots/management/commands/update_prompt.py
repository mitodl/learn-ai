"""Management command for updating a prompt's text content from the command line."""

import sys
from pathlib import Path

from django.conf import settings
from django.core.management import BaseCommand
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client as LangsmithClient
from langsmith.utils import LangSmithError, LangSmithNotFoundError
from open_learning_ai_tutor.prompts import (
    TUTOR_PROMPT_MAPPING,
    assessment_prompt_mapping,
    intent_prompt_mapping,
    prompt_env_key,
)

from ai_chatbots.prompts import SYSTEM_PROMPT_MAPPING
from ai_chatbots.utils import get_django_cache


class Command(BaseCommand):
    """Update a single langsmith prompt key with a new value."""

    help = "Clear the prompt cache."
    cache = get_django_cache()

    def add_arguments(self, parser):
        parser.add_argument(
            "--prompt",
            dest="prompt_name",
            required=True,
            help="Specify the prompt to update",
        )
        parser.add_argument(
            "--content",
            dest="content",
            required=False,
            help="Specify the prompt text",
        )
        parser.add_argument(
            "--contentfile",
            dest="contentfile",
            required=False,
            help="Specify the file containing prompt text",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        """Clear the prompt cache."""

        prompt_name = options["prompt_name"]
        prompt_value = options["content"]
        prompt_file_value = options["contentfile"]

        if prompt_file_value:
            with Path.open(prompt_file_value, "r") as file:
                prompt_value = file.read()

        if not prompt_value and not prompt_file_value:
            for mapping in (
                SYSTEM_PROMPT_MAPPING,
                TUTOR_PROMPT_MAPPING,
                assessment_prompt_mapping,
                intent_prompt_mapping,
            ):
                if prompt_name in mapping:
                    prompt_value = mapping[prompt_name]
                    break
        if not prompt_value:
            self.stderr.write(
                f"Default value for prompt '{prompt_name}' was not found."
            )
            sys.exit(1)

        self.stdout.write(
            self.style.SUCCESS(
                f"Updating prompt '{prompt_name}' with new content: \
                \n\n=======\n{prompt_value}\n=======\n\n"
            )
        )

        cache = get_django_cache()
        prompt_key = prompt_env_key(prompt_name)
        try:
            client = LangsmithClient(api_key=settings.LANGSMITH_API_KEY)
            current_value = client.pull_prompt(prompt_key).messages[0].prompt.template
            if current_value != prompt_value:
                # Prompt exists with different content, make user confirm overwrite
                self.stdout.write(
                    self.style.WARNING(
                        f"Prompt {prompt_key} already exists with different content:\
                        \n\n=======\n{current_value}\n=======\n\n"
                    )
                )
                confirmation = input(
                    f"Do you want to overwrite '{prompt_key}'? (y/n): "
                ).lower()
                if confirmation in ["y", "yes"]:
                    new_prompt_template = ChatPromptTemplate([("system", prompt_value)])
                    client.push_prompt(prompt_key, object=new_prompt_template)
                    cache.delete(prompt_name)
                else:
                    self.stdout.write(self.style.ERROR("Prompt update cancelled."))
                    sys.exit(0)
        except LangSmithNotFoundError:
            # New prompt, push it to LangSmith
            prompt = ChatPromptTemplate([("system", prompt_value)])
            client.push_prompt(prompt_key, object=prompt)
            cache.delete(prompt_name)
        except LangSmithError as le:
            self.stderr.write(f"{le}\n\nError: Please check your Langsmith env values.")
            sys.exit(1)
        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully updated prompt '{prompt_key}' on LangSmith."
            )
        )
        return 0
