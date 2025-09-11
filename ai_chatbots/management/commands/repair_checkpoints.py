"""Management command for adding missing writes to checkpoint metadata"""

from django.core.management import BaseCommand

from ai_chatbots.checkpointers import calculate_writes
from ai_chatbots.models import DjangoCheckpoint
from main.utils import chunks


class Command(BaseCommand):
    """
    Add missing writes to checkpoint metadata. Langgraph 0.6
    no longer includes this in the metadata, but it is useful
    for data platform reporting.
    """

    help = "Add missing writes to checkpoint metadata"

    def batch_assign_writes(self, ids: list[int]):
        """Batch assign writes to checkpoints given a list of ids"""
        checkpoints = DjangoCheckpoint.objects.filter(id__in=ids).only(
            "id", "metadata", "checkpoint"
        )
        for checkpoint in checkpoints:
            checkpoint.metadata["writes"] = calculate_writes(checkpoint.checkpoint)
        DjangoCheckpoint.objects.bulk_update(checkpoints, ["metadata"])

    def handle(self, *args, **options):  # noqa: ARG002
        """Add missing writes to checkpoint metadata"""

        idx = 0
        checkpoint_ids = (
            DjangoCheckpoint.objects.only("id", "metadata")
            .filter(metadata__writes__isnull=True)
            .values_list("id", flat=True)
        )
        self.stdout.write(f"Found {len(checkpoint_ids)} checkpoints to update")
        for batch in chunks(checkpoint_ids, chunk_size=100):
            self.stdout.write(
                f"Processing batch {idx+1} of {len(checkpoint_ids) // 100 + 1}"
            )
            self.batch_assign_writes(batch)
        return 0
