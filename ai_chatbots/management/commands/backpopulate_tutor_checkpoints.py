"""Management command for backpopulating TutorBotOutput checkpoints"""

import json
from uuid import UUID, uuid5

from django.conf import settings
from django.core.management import BaseCommand
from django.db import transaction

from ai_chatbots.api import create_tutor_checkpoints
from ai_chatbots.models import DjangoCheckpoint, TutorBotOutput
from main.utils import chunks


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


def _process_thread_batch(thread_ids_batch, *, overwrite: bool = False) -> int:
    """Process a batch of thread_ids to reduce memory usage"""
    processed = 0
    for thread_id in thread_ids_batch:
        with transaction.atomic():
            if overwrite:
                # Delete existing checkpoints for this thread if overwriting
                DjangoCheckpoint.objects.filter(thread_id=thread_id).delete()
            # If any checkpoints already exist for this thread, skip it
            if DjangoCheckpoint.objects.filter(thread_id=thread_id).exists():
                continue

            # Use iterator to avoid loading all outputs into memory at once
            outputs = (
                TutorBotOutput.objects.filter(thread_id=thread_id)
                .order_by("id")
                .iterator()
            )
            previous_chat_json = None
            # iterate through all outputs instead of just the latest,
            # because some messages may have been truncated.
            for tutorbot_output in outputs:
                # Parse the chat data - handle both string and object formats
                if isinstance(tutorbot_output.chat_json, str):
                    chat_data = json.loads(tutorbot_output.chat_json)
                else:
                    chat_data = tutorbot_output.chat_json

                # Update tutorbot_output with message ids
                chat_data["chat_history"] = add_message_ids(
                    thread_id, tutorbot_output.id, chat_data["chat_history"]
                )
                tutorbot_output.chat_json = json.dumps(chat_data)
                tutorbot_output.save(update_fields=["chat_json"])
                chat_data_str = json.dumps(chat_data)
                create_tutor_checkpoints(
                    thread_id, chat_data_str, previous_chat_json=previous_chat_json
                )
                previous_chat_json = chat_data_str
            processed += 1
    return processed


def convert_tutorbot_to_checkpoints() -> None:
    """
    Add message ids to all TutorBotOutput records and create
    DjangoCheckpoints for new messages in each.
    Memory-efficient version that processes in batches.
    """

    # Use iterator to avoid loading all thread_ids into memory
    thread_ids_qs = (
        TutorBotOutput.objects.only("thread_id")
        .values_list("thread_id", flat=True)
        .distinct()
    )
    # Process in batches using chunks utility
    for thread_ids_chunk in chunks(thread_ids_qs, chunk_size=settings.QUERY_BATCH_SIZE):
        _process_thread_batch(thread_ids_chunk)


class Command(BaseCommand):
    """
    Add missing TutorbotOutput checkpoints.
    """

    help = "Add missing TutorbotOutput checkpoints."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=getattr(settings, "QUERY_BATCH_SIZE", 100),
            help=f"Checkpoint batch size (default: {settings.QUERY_BATCH_SIZE})",
        )
        parser.add_argument(
            "--overwrite",
            dest="force_overwrite",
            action="store_true",
            help="Force regenerate existing TutorBotOutput checkpoints",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        """Add missing writes and state attributes to checkpoint metadata"""

        batch_size = options["batch_size"]
        overwrite = options["force_overwrite"]
        self.stdout.write(
            f"Starting tutor checkpoint backpopulate (batch size: {batch_size})..."
        )

        # Use iterator to avoid loading all thread_ids into memory
        thread_ids_qs = (
            TutorBotOutput.objects.only("thread_id")
            .values_list("thread_id", flat=True)
            .distinct()
        )
        # Process in batches using chunks utility
        total_processed = 0
        for idx, thread_ids_chunk in enumerate(
            chunks(thread_ids_qs, chunk_size=settings.QUERY_BATCH_SIZE)
        ):
            self.stdout.write(
                f"Processing batch {idx + 1} (size: {len(thread_ids_chunk)})..."
            )
            processed = _process_thread_batch(thread_ids_chunk, overwrite=overwrite)
            total_processed += processed

        if total_processed == 0:
            self.stdout.write("No TutorBotOutputs found that need backpopulating")
        else:
            self.stdout.write(f"Completed! Processed {total_processed} TutorBotOutputs")

        return 0
