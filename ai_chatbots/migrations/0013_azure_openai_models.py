import json
from pathlib import Path

from django.db import migrations


def add_azure_models(apps, schema_editor):
    """
    Add the Azure OpenAI deployments as disabled LLMModel rows.

    They are disabled so they stay out of the model list in environments
    without AZURE_OPENAI_ENDPOINT; enable them in the Django admin where
    Azure is configured.
    """
    LLMModel = apps.get_model("ai_chatbots", "LLMModel")
    with Path.open(
        "ai_chatbots/fixtures/migrations/0013_azure_openai_models.json"
    ) as llm_json:
        for llm_model in json.load(llm_json):
            LLMModel.objects.get_or_create(
                litellm_id=llm_model["litellm_id"],
                defaults={
                    "provider": llm_model["provider"],
                    "name": llm_model["name"],
                    "enabled": llm_model["enabled"],
                    "temperature": llm_model.get("temperature", None),
                    "reasoning_effort": llm_model.get("reasoning_effort", ""),
                },
            )


class Migration(migrations.Migration):
    dependencies = [
        ("ai_chatbots", "0012_llm_model_temp_reasoning"),
    ]

    operations = [
        migrations.RunPython(add_azure_models, reverse_code=migrations.RunPython.noop),
    ]
