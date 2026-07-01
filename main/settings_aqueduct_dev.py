"""Opt-in django-aqueduct settings shim for local development.

Select this module with ``DJANGO_SETTINGS_MODULE=main.settings_aqueduct_dev``
to fill in settings missing from the environment via Vault OIDC (through
``DevAqueductSettings``) instead of requiring a local ``.env`` file. Env vars
still take priority over Vault when explicitly set.

``main/settings.py`` is untouched and remains the default in all deployed
environments; this module is purely additive and intended for local dev use.
"""

from django_aqueduct import configure_django_settings

from main.aqueduct_settings import DevAqueductSettings
from main.envs import get_float, get_string
from main.sentry import init_sentry

# initialize Sentry before doing anything else, matching main/settings.py's
# ordering, so we capture any config errors raised while building settings.
init_sentry(
    dsn=get_string("SENTRY_DSN", ""),
    environment=get_string("MITOL_ENVIRONMENT", "dev"),
    version=DevAqueductSettings.model_fields["VERSION"].default,
    log_level=get_string("SENTRY_LOG_LEVEL", "ERROR"),
    traces_sample_rate=get_float("SENTRY_TRACES_SAMPLE_RATE", 0),
    profiles_sample_rate=get_float("SENTRY_PROFILES_SAMPLE_RATE", 0),
)

configure_django_settings(DevAqueductSettings)

# LOGGING is provided by mitol-django-observability (structlog-based, JSON in
# prod), same as main/settings.py -- not modeled as an AqueductSettings
# field, imported directly instead.
from mitol.observability.settings.logging import LOGGING  # noqa: E402, F401
