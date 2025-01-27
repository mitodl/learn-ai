"""Management command to create the checkpoint tables in the Postgres database."""

from django.conf import settings
from django.core.management import BaseCommand
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from ai_chatbots.api import persistence_db


class Command(BaseCommand):
    """Create the langgraph postgres checkpoint tables"""

    help = "Create the langgraph posrgres checkpoint tables"

    def create_checkpointer_tables(self):
        """Create db pool and PostgresSaver instance to create the tables"""
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        with ConnectionPool(
            conninfo=persistence_db(),
            max_size=settings.AI_PERSISTENT_POOL_SIZE,
            kwargs=connection_kwargs,
        ) as pool:
            PostgresSaver(pool).setup()

    def handle(self, *args, **options):  # noqa: ARG002
        """Create the langgraph posrgres checkpoint tables"""
        self.stdout.write("Starting function to create postgres checkpointer tables")
        self.create_checkpointer_tables()
        self.stdout.write("Done!")
