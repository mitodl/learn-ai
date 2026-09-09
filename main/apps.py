import litellm
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

        # litellm's default aiohttp transport caches one ClientSession and,
        # when called from a different event loop than the one it was created
        # on (e.g. Channels' per-consumer async contexts), abandons the old
        # session for GC instead of closing it (see
        # LiteLLMAiohttpTransport._get_valid_client_session in
        # litellm/llms/custom_httpx/aiohttp_transport.py). That leaks open
        # sockets and produces "Unclosed client session" warnings, which is
        # what caused the learn-ai pod memory leak. Fall back to plain httpx,
        # which this app already manages correctly via HTTPClientManager.
        litellm.disable_aiohttp_transport = True
