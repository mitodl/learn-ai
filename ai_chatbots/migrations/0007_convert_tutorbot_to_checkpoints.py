# Generated migration to convert TutorBotOutput to DjangoCheckpoint

import json
from uuid import UUID, uuid5

from django.db import migrations

from ai_chatbots.api import create_tutor_checkpoints, uuid4
from main.utils import chunks, settings


def same_message(msg1, msg2):
    """Check if two messages are the same based on role and content"""
    return msg1.get("role") == msg2.get("role") and msg1.get("content") == msg2.get(
        "content"
    )


def add_message_ids(thread_id: str, output_id: int, messages: list[dict]) -> list[dict]:
    """Add unique IDs to messages that don't have one"""
    for message in messages:
        # Handle both dict and LangChain message objects
        message_id = str(
            uuid5(
                UUID(thread_id), f'{output_id}_{message["type"]}_{message["content"]}'
            )
        )
        if isinstance(message, dict):
            if not message.get("id"):
                message["id"] = message_id
        # # LangChain message object
        elif not hasattr(message, "id") or not message.id:
            message.id = message_id
    return messages


def convert_tutorbot_to_checkpoints(apps, schema_editor):
    """
    Add message ids to all TutorBotOutput records and create
    DjangoCheckpoints for new messages in each.
    Memory-efficient version that processes in batches.
    """

    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")
    DjangoCheckpoint = apps.get_model("ai_chatbots", "DjangoCheckpoint")  # noqa: F841
    UserChatSession = apps.get_model("ai_chatbots", "UserChatSession")  # noqa: F841

    # Use iterator to avoid loading all thread_ids into memory
    thread_ids_qs = (
        TutorBotOutput.objects.only("thread_id")
        .values_list("thread_id", flat=True)
        .distinct()
    )

    # Process in batches using chunks utility
    for thread_ids_chunk in chunks(thread_ids_qs, chunk_size=settings.QUERY_BATCH_SIZE):
        _process_thread_batch(apps, thread_ids_chunk)


def _process_thread_batch(apps, thread_ids_batch):
    """Process a batch of thread_ids to reduce memory usage"""
    from django.db import transaction

    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")

    for thread_id in thread_ids_batch:
        with transaction.atomic():
            # Use iterator to avoid loading all outputs into memory at once
            outputs = (
                TutorBotOutput.objects.filter(thread_id=thread_id)
                .order_by("id")
                .iterator()
            )

            previous_messages = []

            # iterate through all outputs instead of just the latest,
            # because some messages may have been truncated.
            for tutorbot_output in outputs:
                # Parse the chat data - handle both string and object formats
                if isinstance(tutorbot_output.chat_json, str):
                    chat_data = json.loads(tutorbot_output.chat_json)
                else:
                    chat_data = tutorbot_output.chat_json

                current_messages = chat_data.get("chat_history", [])

                if not previous_messages:
                    # First output in thread: all messages are new, assign ids to all
                    current_messages = add_message_ids(
                        thread_id, tutorbot_output.id, current_messages
                    )
                    new_messages = current_messages
                else:
                    # Compare backwards from last message in current to last in previous
                    new_messages = []

                    # Start from the end and work backwards
                    curr_idx = len(current_messages) - 1
                    prev_idx = len(previous_messages) - 1

                    while curr_idx >= 0:
                        current_msg = current_messages[curr_idx]

                        # Check if we have a corresponding previous message to compare
                        if prev_idx >= 0:
                            previous_msg = previous_messages[prev_idx]

                            # If messages are the same, reuse the ID from previous
                            if same_message(current_msg, previous_msg):
                                current_msg["id"] = previous_msg.get("id")
                                prev_idx -= 1
                            else:
                                # Message is different - it's new, assign new ID
                                message_id = str(uuid4())
                                current_msg["id"] = message_id
                                new_messages.append(current_msg)
                        else:
                            # No more previous messages to compare
                            break

                        curr_idx -= 1

                    # Reverse new_messages since we built it backwards
                    new_messages.reverse()

                # Save current_messages as chat_history in the TutorBotOutput
                chat_data["chat_history"] = current_messages
                tutorbot_output.chat_json = json.dumps(chat_data)
                tutorbot_output.save(update_fields=["chat_json"])

                # Create checkpoints for new messages
                if new_messages:
                    create_tutor_checkpoints(
                        thread_id,
                        json.dumps(chat_data),
                        previous_chat_json=json.dumps(
                            {"chat_history": previous_messages}
                        ),
                    )

                # Prepare for next iteration: previous_messages = current_messages
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
