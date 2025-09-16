"""Management command for adding missing writes to checkpoint metadata"""

from django.conf import settings
from django.core.management import BaseCommand

from ai_chatbots.checkpointers import calculate_writes
from ai_chatbots.models import DjangoCheckpoint
from main.utils import chunks


class Command(BaseCommand):
    """
    Add missing writes and state attributes to checkpoint metadata.
    LangGraph 0.6 no longer includes these in the metadata automatically,
    but they are useful for data platform reporting and backward compatibility.
    """

    help = "Add missing writes and state attributes to checkpoint metadata"

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=getattr(settings, "QUERY_BATCH_SIZE", 100),
            help=f"Checkpoint batch size (default: {settings.QUERY_BATCH_SIZE})",
        )

    def batch_assign_writes(self, ids: list[int]):
        """Batch assign writes and state metadata to checkpoints given a list of ids"""
        checkpoints = DjangoCheckpoint.objects.filter(id__in=ids).only(
            "id", "metadata", "checkpoint"
        )
        for checkpoint in checkpoints:
            # Add writes if missing
            if not checkpoint.metadata.get("writes"):
                checkpoint.metadata["writes"] = calculate_writes(checkpoint.checkpoint)

        DjangoCheckpoint.objects.bulk_update(checkpoints, ["metadata"])

    def handle(self, *args, **options):  # noqa: ARG002
        """Add missing writes and state attributes to checkpoint metadata"""

        batch_size = options["batch_size"]
        self.stdout.write(
            f"Starting checkpoint repair process (batch size: {batch_size})..."
        )

        # Use iterator to process checkpoints in batches without loading all IDs
        checkpoint_ids_qs = (
            DjangoCheckpoint.objects.only("id", "metadata")
            .filter(metadata__writes__isnull=True)
            .values_list("id", flat=True)
        )

        total_processed = 0

        for idx, checkpoint_ids_batch in enumerate(
            chunks(checkpoint_ids_qs, chunk_size=batch_size)
        ):
            self.stdout.write(
                f"Processing batch {idx + 1} (size: {len(checkpoint_ids_batch)})..."
            )
            self.batch_assign_writes(checkpoint_ids_batch)
            total_processed += len(checkpoint_ids_batch)

        if total_processed == 0:
            self.stdout.write("No checkpoints found that need repair")
        else:
            self.stdout.write(f"Completed! Processed {total_processed} checkpoints")

        return 0
