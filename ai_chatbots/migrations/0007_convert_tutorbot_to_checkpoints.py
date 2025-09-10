# Generated migration to convert TutorBotOutput to DjangoCheckpoint

import json

from django.db import migrations

from ai_chatbots.api import create_tutor_checkpoints
from ai_chatbots.utils import add_message_ids


def convert_tutorbot_to_checkpoints(apps, schema_editor):
    """Convert all TutorBotOutput records to DjangoCheckpoint format"""
    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")
    DjangoCheckpoint = apps.get_model("ai_chatbots", "DjangoCheckpoint")  # noqa: F841
    UserChatSession = apps.get_model("ai_chatbots", "UserChatSession")  # noqa: F841

    # Group TutorBotOutputs by thread_id and order by id within each group
    thread_ids = TutorBotOutput.objects.values_list("thread_id", flat=True).distinct()

    for thread_id in thread_ids:
        # Get all TutorBotOutputs for this thread, ordered by id
        outputs = TutorBotOutput.objects.filter(thread_id=thread_id).order_by("id")

        # Track message IDs seen so far for this thread
        previous_output = None

        for tutorbot_output in outputs:
            # Parse the chat data - handle both string and object formats
            if isinstance(tutorbot_output.chat_json, str):
                chat_data = json.loads(tutorbot_output.chat_json)
            else:
                # Handle JSONB field that might be stored as string
                chat_data = json.loads(str(tutorbot_output.chat_json).strip('"'))

            messages = add_message_ids(chat_data.get("chat_history", []))
            if not messages:
                continue

            # Update the tutorbot_output with message IDs
            chat_data["chat_history"] = messages
            tutorbot_output.chat_json = chat_data
            tutorbot_output.save(update_fields=["chat_json"])

            previous_chat_json = previous_output.chat_json if previous_output else None
            create_tutor_checkpoints(
                thread_id,
                tutorbot_output.chat_json,
                previous_chat_json=previous_chat_json,
            )
            previous_output = tutorbot_output


def reverse_conversion(apps, schema_editor):
    """Reverse the conversion by deleting converted checkpoints"""
    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")
    DjangoCheckpoint = apps.get_model("ai_chatbots", "DjangoCheckpoint")

    # Get all thread_ids that had TutorBotOutput records
    tutorbot_thread_ids = list(
        TutorBotOutput.objects.values_list("thread_id", flat=True).distinct()
    )

    # Delete the converted checkpoints
    DjangoCheckpoint.objects.filter(thread_id__in=tutorbot_thread_ids).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("ai_chatbots", "0006_tutorbotoutput_edx_module_id"),
    ]

    operations = [
        migrations.RunPython(convert_tutorbot_to_checkpoints, reverse_conversion),
    ]
