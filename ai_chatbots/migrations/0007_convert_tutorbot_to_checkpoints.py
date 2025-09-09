# Generated migration to convert TutorBotOutput to DjangoCheckpoint

import json
from uuid import uuid4

from django.db import migrations, models

from main.utils import now_in_utc


def convert_tutorbot_to_checkpoints(apps, schema_editor):
    """Convert all TutorBotOutput records to DjangoCheckpoint format"""
    TutorBotOutput = apps.get_model("ai_chatbots", "TutorBotOutput")
    DjangoCheckpoint = apps.get_model("ai_chatbots", "DjangoCheckpoint")
    UserChatSession = apps.get_model("ai_chatbots", "UserChatSession")

    # Get only the latest TutorBotOutput for each thread_id
    latest_outputs = TutorBotOutput.objects.filter(
        id__in=TutorBotOutput.objects.values("thread_id")
        .annotate(latest_id=models.Max("id"))
        .values_list("latest_id", flat=True)
    )

    for tutorbot_output in latest_outputs:
        # Get the associated session
        try:
            session = UserChatSession.objects.get(thread_id=tutorbot_output.thread_id)
        except UserChatSession.DoesNotExist:
            continue

        # Parse the chat data - handle both string and object formats
        if isinstance(tutorbot_output.chat_json, str):
            chat_data = json.loads(tutorbot_output.chat_json)
        else:
            # Handle JSONB field that might be stored as string
            chat_data = json.loads(str(tutorbot_output.chat_json).strip('"'))

        messages = chat_data.get("chat_history", [])
        if not messages:
            continue

        # Filter out ToolMessage types - we don't want checkpoints for these
        filtered_messages = [
            msg for msg in messages if msg.get("type") != "ToolMessage"
        ]
        if not filtered_messages:
            continue

        # Create checkpoints for each message with cumulative history
        cumulative_messages = []
        parent_checkpoint_id = None

        for step, message in enumerate(filtered_messages):
            # Generate unique IDs
            checkpoint_id = str(uuid4())
            message_id = str(uuid4())

            # Create message with LangChain format
            langchain_message = {
                "id": ["langchain", "schema", "messages", message["type"]],
                "lc": 1,
                "type": "constructor",
                "kwargs": {
                    "id": message_id,
                    "type": message["type"].lower().replace("message", ""),
                    "content": message["content"],
                },
            }
            cumulative_messages.append(langchain_message)

            # Create checkpoint data with cumulative history
            checkpoint_data = {
                "v": 4,
                "id": checkpoint_id,
                "ts": now_in_utc().isoformat(),
                "pending_sends": [],
                "versions_seen": {
                    "__input__": {},
                    "__start__": {"__start__": step + 1} if step >= 0 else {},
                },
                "channel_values": {
                    "messages": cumulative_messages.copy(),
                    # Preserve tutor-specific data
                    "intent_history": chat_data.get("intent_history"),
                    "assessment_history": chat_data.get("assessment_history"),
                    # Include metadata for reference
                    "tutor_metadata": chat_data.get("metadata", {}),
                    # Add other channel values that might be needed
                    "branch:to:pre_model_hook": None,
                },
                "channel_versions": {
                    "messages": step + 1,
                    "__start__": step + 1 if step >= 0 else 1,
                    "intent_history": 1,
                    "assessment_history": 1,
                    "tutor_metadata": 1,
                    "branch:to:pre_model_hook": step + 1 if step >= 0 else 1,
                },
            }

            # Create metadata with the new message
            if step == 0:
                source = "input"
                writes = (
                    {"__start__": {"messages": [langchain_message]}}
                    if "Human" in message["type"]
                    else None
                )
            else:
                source = "loop"
                writes = (
                    {"agent": {"messages": [langchain_message]}}
                    if "AI" in message["type"]
                    else {"__start__": {"messages": [langchain_message]}}
                )

            metadata = {
                "step": step if step > 0 else -1,
                "source": source,
                "writes": writes,
                "parents": {},
                "thread_id": tutorbot_output.thread_id,
            }

            # Create and save the checkpoint
            DjangoCheckpoint.objects.create(
                session=session,
                thread_id=tutorbot_output.thread_id,
                checkpoint_ns="",
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_checkpoint_id,
                type="msgpack",
                checkpoint=checkpoint_data,
                metadata=metadata,
            )

            parent_checkpoint_id = checkpoint_id


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
