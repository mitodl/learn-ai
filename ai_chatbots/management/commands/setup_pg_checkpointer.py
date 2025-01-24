"""Management command to create the checkpoint tables in the Postgres database."""

from django.core.management import BaseCommand

from ai_chatbots.api import get_postgres_saver


class Command(BaseCommand):
    """Create the langgraph postgres checkpoint tables"""

    help = "Create the langgraph posrgres checkpoint tables"

    def handle(self, *args, **options):  # noqa: ARG002
        """Create the langgraph posrgres checkpoint tables"""

        self.stdout.write("Starting function to create postgres checkpointer tables")
        pg_saver = get_postgres_saver()
        pg_saver.setup()
        self.stdout.write("Done!")
