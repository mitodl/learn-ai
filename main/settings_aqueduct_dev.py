"""Opt-in django-aqueduct settings shim for local development.

Select this module with ``DJANGO_SETTINGS_MODULE=main.settings_aqueduct_dev``
to fill in settings missing from the environment via Vault (through
``DevAqueductSettings``) instead of requiring a local ``.env`` file. The
Vault source is configured entirely from ``VAULT_*`` environment variables
(see the ``DevAqueductSettings`` docstring) and is skipped gracefully when
``VAULT_ADDR`` is unset. Env vars still take priority over Vault when
explicitly set.

``main/settings.py`` is untouched and remains the default in all deployed
environments; this module is purely additive and intended for local dev use.
"""

from django_aqueduct import configure_django_settings

from main.aqueduct_settings import DevAqueductSettings, init_sentry_from_model

configure_django_settings(DevAqueductSettings, pre_configure=init_sentry_from_model)

# LOGGING is provided by mitol-django-observability (structlog-based, JSON in
# prod), same as main/settings.py -- not modeled as an AqueductSettings
# field, imported directly instead.
from mitol.observability.settings.logging import LOGGING  # noqa: E402, F401
