from django.apps import AppConfig


class MainConfig(AppConfig):
    """Main app configuration."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "main"

    def ready(self):
        """Initialize the app"""
        # Initialize features
        from main import features
        from main.opik_keycloak_auth import configure_opik_keycloak_auth

        features.configure()
        # Must run before any Opik client/tracer is created so SDK requests
        # authenticate against the Keycloak-fronted Opik gateway.
        configure_opik_keycloak_auth()
