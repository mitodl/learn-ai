"""Anonymize duplicate sessions per user per identical updated_on timestamp.

For each user, group that user's UserChatSession rows by their exact
updated_on value. If any group has count > 1, set the user field to
NULL for those sessions.
"""

from collections import defaultdict
from collections.abc import Iterable

from django.conf import settings
from django.core.management import BaseCommand
from django.db import transaction
from django.db.models import Count

from ai_chatbots.models import UserChatSession
from main.utils import chunks


def duplicate_updated_on_sessions():
    """Return queryset of (user_id, updated_on, dup_count) where dup_count > 1."""
    return (
        UserChatSession.objects.exclude(user__isnull=True)
        .values("user_id", "updated_on")
        .annotate(dup_count=Count("id"))
        .filter(dup_count__gt=1)
        .order_by("user_id")
    )


def fetch_session_ids_per_dt(user_id, updated_on) -> list[int]:
    """Fetch ids of sessions for a user with a specific updated_on timestamp."""
    return list(
        UserChatSession.objects.filter(
            user_id=user_id, updated_on=updated_on
        ).values_list("id", flat=True)
    )


def anonymize_sessions(session_ids: Iterable[int]) -> int:
    """Null out user for the given session ids. Returns number updated."""
    if not session_ids:
        return 0
    # Use update to avoid calling save() per instance
    return UserChatSession.objects.filter(id__in=list(session_ids)).update(user=None)


class Command(BaseCommand):
    help = (
        "Anonymize (set user=None) UserChatSession rows where a user has more than one "
        "session sharing the identical updated_on timestamp."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=settings.QUERY_BATCH_SIZE,
            help="Number of duplicate (user, updated_on) groups to process per batch.",
        )
        parser.add_argument(
            "--commit",
            action="store_true",
            help="Commit changes to the database. Without this flag it's a dry run.",
        )

    def handle(self, *args, **options):  # noqa: ARG002
        batch_size = options["batch_size"]
        commit = options["commit"]

        duplicate_updates = duplicate_updated_on_sessions()

        total_groups = duplicate_updates.count()
        self.stdout.write(
            f"Found {total_groups} duplicate (user, updated_on) groups to process"
        )

        if total_groups == 0:
            self.stdout.write("Nothing to do.")
            return 0

        total_sessions = 0
        total_detached = 0

        # Process in batches of groups
        for idx, group_batch in enumerate(
            chunks(duplicate_updates.iterator(), chunk_size=batch_size), start=1
        ):
            # Group by user for clearer logging (optional)
            groups_by_user: dict[int, list[dict]] = defaultdict(list)
            group_list = list(group_batch)
            for group in group_list:
                groups_by_user[group["user_id"]].append(group)

            with transaction.atomic():
                for user_id, groups in groups_by_user.items():
                    for group in groups:
                        updated_on = group["updated_on"]
                        session_ids = fetch_session_ids_per_dt(user_id, updated_on)
                        total_sessions += len(session_ids)
                        if commit:
                            detached = anonymize_sessions(session_ids)
                            total_detached += detached
                        else:
                            # Dry run: pretend; don't modify
                            total_detached += len(session_ids)
                self.stdout.write(
                    f"Processed batch {idx}: cumulative_sessions={total_sessions}"
                )

        if commit:
            self.stdout.write(
                self.style.SUCCESS(f"Anonymized {total_detached} sessions")
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"DRY RUN: Anonymize {total_detached} sessions"
                    "Re-run with --commit to apply."
                )
            )
        return 0
