import litellm
from django.apps import AppConfig


class MainConfig(AppConfig):
    """Main app configuration."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "main"

    def ready(self):
        """Initialize the app"""
        import ol_llm
        from django.conf import settings

        from main import features

        # Initialize features
        features.configure()
        # Before anything uses the Opik SDK, which reads ol-llm's Keycloak auth
        # hook only when it builds its client.
        ol_llm.configure(ol_llm.Settings(environment=settings.OL_LLM_ENVIRONMENT))
        ol_llm.configure_tracing(
            None if settings.OPIK_URL_OVERRIDE else ol_llm.TracingConfig(enabled=False)
        )

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
