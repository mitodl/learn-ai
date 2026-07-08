"""Opt-in django-aqueduct settings shim.

Select this module with ``DJANGO_SETTINGS_MODULE=main.settings_aqueduct`` to
validate/load settings through the typed ``AqueductSettings`` model in
``main/aqueduct_settings.py`` instead of ``main/settings.py``. It reads the
same environment variables as ``main/settings.py`` -- no separate
configuration is required.

Sentry is initialized through the ``pre_configure`` hook: after the model has
resolved values through its source chain (so a Vault-supplied SENTRY_DSN
works) but before the settings are injected, matching main/settings.py's
init-Sentry-first ordering.

``main/settings.py`` is untouched and remains the default in all deployed
environments; this module is purely additive.
"""

from django_aqueduct import configure_django_settings

from main.aqueduct_settings import AqueductSettings, init_sentry_from_model

configure_django_settings(AqueductSettings, pre_configure=init_sentry_from_model)

# LOGGING is provided by mitol-django-observability (structlog-based, JSON in
# prod), same as main/settings.py -- not modeled as an AqueductSettings
# field, imported directly instead.
from mitol.observability.settings.logging import LOGGING  # noqa: E402, F401
