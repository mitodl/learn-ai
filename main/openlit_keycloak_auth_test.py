"""Tests for Keycloak client-credentials auth for OpenLIT's OTLP export."""

from types import SimpleNamespace

import pytest
import requests
import responses

from main import openlit_keycloak_auth
from main.openlit_keycloak_auth import (
    KeycloakClientCredentialsAuth,
    OpenLITScopeLogProcessor,
    OpenLITScopeMetricExporter,
    OpenLITScopeSpanProcessor,
    _install_authed_providers,
    configure_openlit,
    is_openlit_configured,
)

TOKEN_URL = "https://sso.example.edu/realms/olapps/protocol/openid-connect/token"  # noqa: S105
OTLP_ENDPOINT = "https://openlit-ci.example.edu"


@pytest.fixture
def openlit_settings(settings):
    """Set all required OpenLIT settings."""
    settings.OPENLIT_OTLP_ENDPOINT = OTLP_ENDPOINT
    settings.OPENLIT_KEYCLOAK_TOKEN_URL = TOKEN_URL
    settings.OPENLIT_KEYCLOAK_CLIENT_ID = "ol-openlit-client"
    settings.OPENLIT_KEYCLOAK_CLIENT_SECRET = "secret"  # noqa: S105
    settings.OPENLIT_KEYCLOAK_SCOPE = None
    return settings


def make_auth(scope=None):
    """Build an auth instance pointed at the canned token endpoint."""
    return KeycloakClientCredentialsAuth(
        token_url=TOKEN_URL,
        client_id="ol-openlit-client",
        client_secret="secret",  # noqa: S106
        scope=scope,
    )


def add_token_response(mocked_responses, token, expires_in=300):
    """Register a canned Keycloak token endpoint response."""
    mocked_responses.add(
        responses.POST,
        TOKEN_URL,
        json={"access_token": token, "expires_in": expires_in},
    )


def otlp_request():
    """Build a prepared OTLP export request for the auth to sign."""
    return requests.Request("POST", f"{OTLP_ENDPOINT}/v1/traces").prepare()


def test_auth_fetches_and_caches_token(mocked_responses):
    """The first request fetches a token; later requests reuse the cache."""
    add_token_response(mocked_responses, "token-1")
    auth = make_auth()

    first = auth(otlp_request())
    second = auth(otlp_request())

    assert first.headers["Authorization"] == "Bearer token-1"
    assert second.headers["Authorization"] == "Bearer token-1"
    assert len(mocked_responses.calls) == 1


def test_auth_refreshes_expired_token(mocked_responses):
    """An expired cached token is replaced before the request is sent."""
    add_token_response(mocked_responses, "token-1")
    add_token_response(mocked_responses, "token-2")
    auth = make_auth()

    auth(otlp_request())
    auth._expires_at = 0.0  # noqa: SLF001 - simulate expiry
    request = auth(otlp_request())

    assert request.headers["Authorization"] == "Bearer token-2"
    assert len(mocked_responses.calls) == 2


def test_auth_scope_included_when_set(mocked_responses):
    """The optional scope is sent to the token endpoint."""
    add_token_response(mocked_responses, "token-1")
    auth = make_auth(scope="openlit")

    auth(otlp_request())

    assert "scope=openlit" in mocked_responses.calls[0].request.body


def test_token_endpoint_error_propagates(mocked_responses):
    """A failing token endpoint raises instead of caching a bad token."""
    mocked_responses.add(responses.POST, TOKEN_URL, json={"error": "boom"}, status=500)
    auth = make_auth()

    with pytest.raises(requests.HTTPError):
        auth._get_token()  # noqa: SLF001


def test_is_openlit_configured(settings):
    """is_openlit_configured requires endpoint, token URL, client id and secret."""
    settings.OPENLIT_OTLP_ENDPOINT = None
    settings.OPENLIT_KEYCLOAK_TOKEN_URL = None
    settings.OPENLIT_KEYCLOAK_CLIENT_SECRET = None
    assert is_openlit_configured() is False


def test_is_openlit_configured_true(openlit_settings):
    """is_openlit_configured is True when everything is set."""
    assert is_openlit_configured() is True


def test_configure_skips_when_unconfigured(settings, mocker):
    """OpenLIT is not initialized when its settings are absent."""
    settings.OPENLIT_OTLP_ENDPOINT = None
    settings.OPENLIT_KEYCLOAK_TOKEN_URL = None
    settings.OPENLIT_KEYCLOAK_CLIENT_SECRET = None
    mocker.patch.object(openlit_keycloak_auth, "_configured", new=False)
    init = mocker.patch("openlit.init")
    install = mocker.patch.object(openlit_keycloak_auth, "_install_authed_providers")

    assert configure_openlit() is False
    init.assert_not_called()
    install.assert_not_called()


def test_configure_initializes_once(openlit_settings, mocker):
    """Providers are installed and openlit.init runs exactly once, unauthed-endpoint-free."""
    mocker.patch.object(openlit_keycloak_auth, "_configured", new=False)
    init = mocker.patch("openlit.init")
    install = mocker.patch.object(openlit_keycloak_auth, "_install_authed_providers")

    assert configure_openlit() is True
    assert configure_openlit() is True

    install.assert_called_once()
    auth = install.call_args[0][0]
    assert isinstance(auth, KeycloakClientCredentialsAuth)
    assert auth._token_url == TOKEN_URL  # noqa: SLF001

    init.assert_called_once()
    kwargs = init.call_args.kwargs
    assert kwargs["service_name"] == openlit_settings.OPENTELEMETRY_SERVICE_NAME
    assert kwargs["environment"] == openlit_settings.ENVIRONMENT
    # A static endpoint/headers pair would hijack OTEL_EXPORTER_OTLP_* for the
    # whole process; the authed providers are the only export path.
    assert "otlp_endpoint" not in kwargs
    assert "otlp_headers" not in kwargs


@pytest.fixture
def provider_setters(mocker):
    """Spy on the global OTel provider setters; shut down whatever was built."""
    setters = SimpleNamespace(
        tracer=mocker.patch("opentelemetry.trace.set_tracer_provider"),
        meter=mocker.patch("opentelemetry.metrics.set_meter_provider"),
        logger=mocker.patch("opentelemetry._logs.set_logger_provider"),
    )
    yield setters
    for setter in (setters.tracer, setters.meter, setters.logger):
        for call in setter.call_args_list:
            call.args[0].shutdown()


def test_install_reuses_existing_tracer_provider(
    openlit_settings, provider_setters, mocker
):
    """An SDK TracerProvider set by mitol-django-observability is extended, not replaced."""
    from opentelemetry.sdk.trace import TracerProvider

    existing = TracerProvider()
    mocker.patch("opentelemetry.trace.get_tracer_provider", return_value=existing)
    auth = make_auth()

    _install_authed_providers(auth)

    provider_setters.tracer.assert_not_called()
    processors = existing._active_span_processor._span_processors  # noqa: SLF001
    assert len(processors) == 1
    assert isinstance(processors[0], OpenLITScopeSpanProcessor)
    exporter = processors[0]._inner.span_exporter  # noqa: SLF001
    assert exporter._endpoint == f"{OTLP_ENDPOINT}/v1/traces"  # noqa: SLF001
    assert exporter._session.auth is auth  # noqa: SLF001
    provider_setters.meter.assert_called_once()
    provider_setters.logger.assert_called_once()
    existing.shutdown()


def test_install_creates_tracer_provider_when_absent(
    openlit_settings, provider_setters, mocker
):
    """Without a configured SDK provider (proxy default), one is created and set."""
    from opentelemetry.trace import ProxyTracerProvider

    mocker.patch(
        "opentelemetry.trace.get_tracer_provider", return_value=ProxyTracerProvider()
    )
    auth = make_auth()

    _install_authed_providers(auth)

    provider_setters.tracer.assert_called_once()
    provider = provider_setters.tracer.call_args[0][0]
    processors = provider._active_span_processor._span_processors  # noqa: SLF001
    assert isinstance(processors[0], OpenLITScopeSpanProcessor)
    exporter = processors[0]._inner.span_exporter  # noqa: SLF001
    assert exporter._endpoint == f"{OTLP_ENDPOINT}/v1/traces"  # noqa: SLF001


def test_span_processor_forwards_only_openlit_scopes(mocker):
    """Spans from non-openlit tracers (celery, django, ...) are not exported."""
    from opentelemetry.sdk.trace import TracerProvider

    inner = mocker.Mock()
    provider = TracerProvider()
    provider.add_span_processor(OpenLITScopeSpanProcessor(inner))

    tracer = provider.get_tracer("openlit.instrumentation.langchain")
    with tracer.start_as_current_span("chat gpt-4"):
        pass
    with provider.get_tracer(
        "opentelemetry.instrumentation.celery"
    ).start_as_current_span("celery task"):
        pass

    assert inner.on_end.call_count == 1
    assert inner.on_end.call_args[0][0].name == "chat gpt-4"
    provider.shutdown()


def test_log_processor_forwards_only_openlit_scopes(mocker):
    """Log records from non-openlit loggers are not exported."""
    inner = mocker.Mock()
    processor = OpenLITScopeLogProcessor(inner)
    openlit_record = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="openlit.otel.events")
    )
    other_record = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="my.app.logger")
    )

    processor.on_emit(openlit_record)
    processor.on_emit(other_record)

    inner.on_emit.assert_called_once_with(openlit_record)


def _scope_metrics(name):
    """Build an empty ScopeMetrics for the named instrumentation scope."""
    from opentelemetry.sdk.metrics.export import ScopeMetrics
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    return ScopeMetrics(scope=InstrumentationScope(name), metrics=[], schema_url="")


def _metrics_data(*scope_names):
    """Build a MetricsData carrying one ScopeMetrics per named scope."""
    from opentelemetry.sdk.metrics.export import MetricsData, ResourceMetrics
    from opentelemetry.sdk.resources import Resource

    return MetricsData(
        resource_metrics=[
            ResourceMetrics(
                resource=Resource.create({}),
                scope_metrics=[_scope_metrics(name) for name in scope_names],
                schema_url="",
            )
        ]
    )


def test_metric_exporter_prunes_non_openlit_scopes(mocker):
    """Only openlit-scoped metrics reach the underlying OTLP exporter."""
    from opentelemetry.sdk.metrics.export import MetricExportResult

    parent_export = mocker.patch(
        "main.openlit_keycloak_auth.OTLPMetricExporter.export",
        return_value=MetricExportResult.SUCCESS,
    )
    exporter = OpenLITScopeMetricExporter(endpoint=f"{OTLP_ENDPOINT}/v1/metrics")

    result = exporter.export(
        _metrics_data("openlit.otel.metrics", "opentelemetry.instrumentation.django")
    )

    assert result == MetricExportResult.SUCCESS
    exported = parent_export.call_args[0][0]
    scope_names = [
        scope_metric.scope.name
        for resource_metric in exported.resource_metrics
        for scope_metric in resource_metric.scope_metrics
    ]
    assert scope_names == ["openlit.otel.metrics"]


def test_metric_exporter_skips_export_without_openlit_metrics(mocker):
    """Nothing is sent at all when no openlit-scoped metrics were collected."""
    from opentelemetry.sdk.metrics.export import MetricExportResult

    parent_export = mocker.patch("main.openlit_keycloak_auth.OTLPMetricExporter.export")
    exporter = OpenLITScopeMetricExporter(endpoint=f"{OTLP_ENDPOINT}/v1/metrics")

    result = exporter.export(_metrics_data("opentelemetry.instrumentation.celery"))

    assert result == MetricExportResult.SUCCESS
    parent_export.assert_not_called()
