from django.apps import AppConfig


class MainConfig(AppConfig):
    """Main app configuration."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "main"

    def ready(self):
        """Initialize the app"""
        # Initialize features
        from main import features

        features.configure()

        # Initialize OpenTelemetry
        from main.telemetry import configure_opentelemetry

        configure_opentelemetry()
