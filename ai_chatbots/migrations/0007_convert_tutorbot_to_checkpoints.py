# Generated migration to convert TutorBotOutput to DjangoCheckpoint

import json

from django.db import migrations

from ai_chatbots.api import create_tutor_checkpoints
from ai_chatbots.utils import add_message_ids


def same_message(msg1, msg2):
    """Check if two messages are the same based on role and content"""
    return msg1.get("role") == msg2.get("role") and msg1.get("content") == msg2.get(
        "content"
    )


def convert_tutorbot_to_checkpoints(apps, schema_editor):
    """
    Add message ids to all TutorBotOutput records and create
    DjangoCheckpoints for new messages in each.
    """
    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")
    DjangoCheckpoint = apps.get_model("ai_chatbots", "DjangoCheckpoint")  # noqa: F841
    UserChatSession = apps.get_model("ai_chatbots", "UserChatSession")  # noqa: F841

    # Group TutorBotOutputs by thread_id and order by id within each group
    thread_ids = TutorBotOutput.objects.values_list("thread_id", flat=True).distinct()

    for thread_id in thread_ids:
        # Get all TutorBotOutputs for this thread, ordered by id
        outputs = TutorBotOutput.objects.filter(thread_id=thread_id).order_by("id")
        previous_messages = []

        for tutorbot_output in outputs:
            # Parse the chat data - handle both string and object formats
            if isinstance(tutorbot_output.chat_json, str):
                chat_data = json.loads(tutorbot_output.chat_json)
            else:
                # Handle JSONB field
                chat_data = json.loads(str(tutorbot_output.chat_json).strip('"'))

            # Determine which messages are new and need to be checkpointed
            current_messages = chat_data.get("chat_history", [])
            final_match_idx = -1
            for previous_message in previous_messages:
                for current_idx, current_message in enumerate(current_messages):
                    if (
                        same_message(previous_message, current_message)
                        and current_idx > final_match_idx
                    ):
                        final_match_idx = current_idx
                        break
            new_messages = (
                current_messages[final_match_idx + 1 :]
                if final_match_idx != -1
                else current_messages
            )
            if not new_messages:
                continue

            # Update tutorbot_output with message ids
            chat_data["chat_history"] = add_message_ids(
                thread_id, tutorbot_output.id, new_messages
            )
            tutorbot_output.chat_json = json.dumps(chat_data)
            tutorbot_output.save(update_fields=["chat_json"])
            chat_data_str = json.dumps(chat_data)

            # Create a checkpoint for each new message
            create_tutor_checkpoints(
                thread_id,
                chat_data_str,
            )
            previous_messages = current_messages


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
