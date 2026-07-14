"""Keycloak client-credentials auth for OpenLIT's OTLP export path.

Our OpenLIT installation has no native auth; Keycloak OIDC sits in front of it
at the APISIX gateway, which validates a JWT bearer token on every OTLP ingest
request (``/v1/traces``, ``/v1/metrics``, ``/v1/logs``, HTTP/protobuf only).
OpenLIT does not own an HTTP client — ``openlit.init()`` builds stock
OpenTelemetry OTLP/HTTP exporters, whose headers are read once at
construction, so a static ``Authorization`` header would go stale within
minutes. Instead, each exporter is given a ``requests.Session`` whose ``auth``
fetches and refreshes a Keycloak access token via the client-credentials
grant.

See
https://github.com/mitodl/ol-infrastructure/blob/main/src/ol_infrastructure/applications/openlit/OPENLIT_SDK_KEYCLOAK_AUTH.md

``openlit.init()`` reuses pre-existing global OTel SDK providers for all three
signals and only builds its own (unauthenticated, env-var-driven) exporters
when none are configured. ``configure_openlit()`` therefore installs authed
providers first, then calls ``openlit.init()``:

* Tracer: ``mitol-django-observability`` may already have installed the global
  ``TracerProvider`` (when ``OPENTELEMETRY_ENDPOINT`` is set or ``DEBUG``); in
  that case an authed OpenLIT span processor is added to it. Otherwise a new
  provider is installed.
* Meter/Logger: nothing else in this app sets these, so authed providers are
  installed outright.

Sharing the global providers means every instrumentor in the process (celery,
django, redis, psycopg, ...) writes to them, not just OpenLIT's LLM
instrumentors. All OpenLIT tracers/meters/loggers are scoped ``openlit.*``
(``openlit.instrumentation.langchain``, ``openlit.otel.metrics``, ...), so the
OpenLIT export path filters on instrumentation scope: only openlit-scoped
telemetry is shipped to OpenLIT, while the shared spans still flow unfiltered
to the mitol-django-observability exporter.

``configure_openlit()`` is called once at startup from
``main.apps.MainConfig.ready`` — after ``ObservabilityConfig.ready`` (app
order in ``INSTALLED_APPS``), so the tracer-provider check above sees the
final state. Do NOT pass ``otlp_endpoint``/``otlp_headers`` to
``openlit.init()``: with providers pre-installed they are ignored at best,
and ``otlp_endpoint`` writes ``OTEL_EXPORTER_OTLP_ENDPOINT`` into the process
environment, hijacking every other OTel exporter in the process.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import TYPE_CHECKING

import requests
from django.conf import settings
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LogRecordProcessor
from opentelemetry.sdk.metrics.export import MetricExportResult, MetricsData
from opentelemetry.sdk.trace import SpanProcessor

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk._logs import ReadWriteLogRecord
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.trace import ReadableSpan, Span
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

log = logging.getLogger(__name__)

# Instrumentation-scope prefix shared by every tracer/meter/logger OpenLIT
# creates (openlit.otel.tracing, openlit.instrumentation.langchain, ...).
_OPENLIT_SCOPE_PREFIX = "openlit"

# Renew this many seconds before the server-stated expiry to avoid edge races.
# A batch that does go out with a stale token is dropped by the exporter (401
# is non-retryable in OTLP), which is why the skew matters.
_REFRESH_SKEW_SECONDS = 30
# Bound the token request so a hung Keycloak cannot wedge exporter threads.
_TOKEN_REQUEST_TIMEOUT_SECONDS = 10

_configured = False


class KeycloakClientCredentialsAuth(requests.auth.AuthBase):
    """requests auth that injects a Keycloak access token via client-credentials.

    A single instance is shared across every OTLP exporter session (each
    exporter flushes from its own background thread, so each gets its own
    ``requests.Session``, but all sessions share this auth). The cached token
    is refreshed proactively before expiry.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: str | None = None,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        # Dedicated session for the token endpoint so token requests never
        # recurse through an (authed) exporter session.
        self._token_session = requests.Session()

        self._lock = threading.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0  # monotonic-clock deadline

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Attach a bearer token to the outgoing OTLP request."""
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        return request

    def _get_token(self) -> str:
        with self._lock:
            if self._access_token is None or time.monotonic() >= self._expires_at:
                self._refresh_locked()
            return self._access_token

    def _refresh_locked(self) -> None:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scope:
            data["scope"] = self._scope

        resp = self._token_session.post(
            self._token_url, data=data, timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        payload = resp.json()

        self._access_token = payload["access_token"]
        expires_in = float(payload.get("expires_in", 60))
        self._expires_at = time.monotonic() + max(
            0.0, expires_in - _REFRESH_SKEW_SECONDS
        )


def _authed_session(auth: KeycloakClientCredentialsAuth) -> requests.Session:
    session = requests.Session()
    session.auth = auth
    return session


def _is_openlit_scope(scope: InstrumentationScope | None) -> bool:
    return scope is not None and (scope.name or "").startswith(_OPENLIT_SCOPE_PREFIX)


class OpenLITScopeSpanProcessor(SpanProcessor):
    """Forward only openlit-instrumented spans to the wrapped processor.

    The OpenLIT span processor hangs off the shared global ``TracerProvider``,
    so without this filter every span in the process (celery tasks, django
    requests, redis polling, ...) would be exported to OpenLIT alongside the
    LLM spans.
    """

    def __init__(self, inner: BatchSpanProcessor) -> None:
        self._inner = inner

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """No-op; filtering and batching happen at span end."""

    def on_end(self, span: ReadableSpan) -> None:
        """Queue the span for export only if an openlit tracer created it."""
        if _is_openlit_scope(span.instrumentation_scope):
            self._inner.on_end(span)

    def shutdown(self) -> None:
        """Shut down the wrapped processor."""
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush the wrapped processor."""
        return self._inner.force_flush(timeout_millis)


class OpenLITScopeLogProcessor(LogRecordProcessor):
    """Forward only openlit-emitted log records (GenAI events) for export."""

    def __init__(self, inner: BatchLogRecordProcessor) -> None:
        self._inner = inner

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        """Queue the record for export only if an openlit logger emitted it."""
        if _is_openlit_scope(log_record.instrumentation_scope):
            self._inner.on_emit(log_record)

    def shutdown(self) -> None:
        """Shut down the wrapped processor."""
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush the wrapped processor."""
        return self._inner.force_flush(timeout_millis)


class OpenLITScopeMetricExporter(OTLPMetricExporter):
    """OTLP metric exporter that ships only openlit-scoped metrics.

    The global ``MeterProvider`` installed for OpenLIT also collects metrics
    recorded by the OTel framework instrumentors (django, celery, ...); those
    are pruned here so OpenLIT receives only its own GenAI metrics.
    """

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float | None = 10_000,
        **kwargs,
    ) -> MetricExportResult:
        """Export openlit-scoped metrics, dropping everything else."""
        resource_metrics = []
        for rm in metrics_data.resource_metrics:
            kept = [sm for sm in rm.scope_metrics if _is_openlit_scope(sm.scope)]
            if kept:
                resource_metrics.append(dataclasses.replace(rm, scope_metrics=kept))
        if not resource_metrics:
            return MetricExportResult.SUCCESS
        return super().export(
            MetricsData(resource_metrics=resource_metrics), timeout_millis, **kwargs
        )


def is_openlit_configured() -> bool:
    """Return True if all settings required for OpenLIT export are present."""
    return all(
        (
            settings.OPENLIT_OTLP_ENDPOINT,
            settings.OPENLIT_KEYCLOAK_TOKEN_URL,
            settings.OPENLIT_KEYCLOAK_CLIENT_ID,
            settings.OPENLIT_KEYCLOAK_CLIENT_SECRET,
        )
    )


def _install_authed_providers(auth: KeycloakClientCredentialsAuth) -> None:
    """Install OTel SDK providers whose OTLP exporters carry Keycloak auth.

    Exporters get explicit per-signal endpoints; the ``OTEL_EXPORTER_OTLP_*``
    environment variables stay untouched for other OTel components (the
    mitol-django-observability exporter reads them).
    """
    from opentelemetry import _logs, metrics, trace
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = settings.OPENLIT_OTLP_ENDPOINT.rstrip("/")

    # Resource attributes OpenLIT would otherwise stamp itself; it reuses our
    # providers, so we must provide them.
    resource = Resource.create(
        attributes={
            "service.name": settings.OPENTELEMETRY_SERVICE_NAME,
            "deployment.environment": settings.ENVIRONMENT,
            "telemetry.sdk.name": "openlit",
        }
    )

    span_processor = OpenLITScopeSpanProcessor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{endpoint}/v1/traces", session=_authed_session(auth)
            )
        )
    )
    existing_tracer_provider = trace.get_tracer_provider()
    if isinstance(existing_tracer_provider, TracerProvider):
        # mitol-django-observability already owns the global provider; add
        # OpenLIT as an additional (openlit-scope-only) export destination.
        existing_tracer_provider.add_span_processor(span_processor)
    else:
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)

    metrics.set_meter_provider(
        MeterProvider(
            resource=resource,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OpenLITScopeMetricExporter(
                        endpoint=f"{endpoint}/v1/metrics",
                        session=_authed_session(auth),
                    )
                )
            ],
        )
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        OpenLITScopeLogProcessor(
            BatchLogRecordProcessor(
                OTLPLogExporter(
                    endpoint=f"{endpoint}/v1/logs", session=_authed_session(auth)
                )
            )
        )
    )
    _logs.set_logger_provider(logger_provider)


def configure_openlit() -> bool:
    """Install Keycloak-authed OTel providers and initialize OpenLIT.

    Must run before any LLM library call worth tracing; the OpenLIT
    instrumentors patch the libraries at ``openlit.init()`` time.

    Returns True if OpenLIT was initialized (or already was), False if the
    OpenLIT settings are absent.
    """
    global _configured  # noqa: PLW0603

    if not is_openlit_configured():
        log.debug("OpenLIT settings not configured, skipping instrumentation")
        return False
    if _configured:
        return True

    import openlit

    auth = KeycloakClientCredentialsAuth(
        token_url=settings.OPENLIT_KEYCLOAK_TOKEN_URL,
        client_id=settings.OPENLIT_KEYCLOAK_CLIENT_ID,
        client_secret=settings.OPENLIT_KEYCLOAK_CLIENT_SECRET,
        scope=settings.OPENLIT_KEYCLOAK_SCOPE,
    )
    _install_authed_providers(auth)

    openlit.init(
        service_name=settings.OPENTELEMETRY_SERVICE_NAME,
        environment=settings.ENVIRONMENT,
        capture_message_content=settings.OPENLIT_CAPTURE_MESSAGE_CONTENT,
        disabled_instrumentors=settings.OPENLIT_DISABLED_INSTRUMENTORS or None,
    )
    _configured = True
    log.info("OpenLIT initialized with Keycloak-authed OTLP export")
    return True
