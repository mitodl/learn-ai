"""Tests for the main app config"""

import ol_llm
import pytest
from django.apps import apps


@pytest.mark.parametrize("opik_url", ["https://opik.example.com/api/", ""])
def test_ready_traces_only_with_an_opik_url(mocker, settings, opik_url):
    """Without OPIK_URL_OVERRIDE the Opik SDK would send traces to Comet's cloud."""
    mocker.patch.object(ol_llm, "configure")
    configure_tracing = mocker.patch.object(ol_llm, "configure_tracing")
    settings.OPIK_URL_OVERRIDE = opik_url

    apps.get_app_config("main").ready()

    configure_tracing.assert_called_once_with(
        None if opik_url else ol_llm.TracingConfig(enabled=False)
    )
