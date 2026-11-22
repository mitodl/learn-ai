import json
from pathlib import Path

from django.db import migrations


def add_gpt_models(apps, schema_editor):
    """
    Add gpt-5.4, gpt-5.4-mini, gpt-6-luna and gpt-6-sol.

    reasoning_effort is "none": gpt-6-luna and gpt-6-sol default to "medium",
    and Chat Completions only allows their tool calls with reasoning off.
    The azure/ rows are disabled, like those from 0013, until Azure is
    configured where they are used.
    """
    LLMModel = apps.get_model("ai_chatbots", "LLMModel")
    with Path.open(
        "ai_chatbots/fixtures/migrations/0016_gpt_5_4_and_6_models.json"
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
        ("ai_chatbots", "0015_support_ticket_email"),
    ]

    operations = [
        migrations.RunPython(add_gpt_models, reverse_code=migrations.RunPython.noop),
    ]
