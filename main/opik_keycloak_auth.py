"""Keycloak client-credentials auth for the Opik SDK.

Our Opik installation has no native auth; Keycloak OIDC sits in front of it at
the APISIX gateway, which validates a JWT bearer token on every ``/api/*``
request. The Opik Python SDK reuses a single ``httpx.Client`` for all traffic,
and exposes ``opik.hooks.add_httpx_client_hook`` as a supported extension
point, so we attach an ``httpx.Auth`` that fetches and refreshes a Keycloak
access token via the client-credentials grant.

See
https://github.com/mitodl/ol-infrastructure/blob/main/src/ol_infrastructure/applications/opik/OPIK_SDK_KEYCLOAK_AUTH.md

``configure_opik_keycloak_auth()`` is called once at startup from
``main.apps.MainConfig.ready``, before any Opik client or tracer is
instantiated. Do NOT set ``OPIK_API_KEY``; the SDK would stamp a static
``Authorization`` header that conflicts with this flow.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import httpx
from django.conf import settings

if TYPE_CHECKING:
    from collections.abc import Generator

log = logging.getLogger(__name__)

# Renew this many seconds before the server-stated expiry to avoid edge races.
_REFRESH_SKEW_SECONDS = 30
# Bound the token request so a hung Keycloak cannot wedge SDK background workers.
_TOKEN_REQUEST_TIMEOUT_SECONDS = 10

_configured = False


class KeycloakClientCredentialsAuth(httpx.Auth):
    """httpx auth flow that injects a Keycloak access token via client-credentials.

    A single instance is shared across every request the SDK makes. The cached
    token is refreshed proactively before expiry and reactively on a 401.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: str | None = None,
        verify: bool | str = True,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        # Dedicated client for the token endpoint so we never recurse through the
        # SDK's own (authed) client. `verify` mirrors the SDK's TLS setting.
        self._token_client = httpx.Client(
            timeout=_TOKEN_REQUEST_TIMEOUT_SECONDS, verify=verify
        )

        self._lock = threading.Lock()
        self._access_token: str | None = None
        self._expires_at: float = 0.0  # monotonic-clock deadline

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Attach a bearer token; on a 401 refresh it and retry exactly once."""
        request.headers["Authorization"] = f"Bearer {self._get_token()}"
        response = yield request

        if response.status_code == httpx.codes.UNAUTHORIZED:
            # Token may have been revoked or expired between fetch and use.
            response.close()
            request.headers["Authorization"] = (
                f"Bearer {self._get_token(force_refresh=True)}"
            )
            yield request

    def _get_token(self, *, force_refresh: bool = False) -> str:
        with self._lock:
            now = time.monotonic()
            if force_refresh or self._access_token is None or now >= self._expires_at:
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

        resp = self._token_client.post(self._token_url, data=data)
        resp.raise_for_status()
        payload = resp.json()

        self._access_token = payload["access_token"]
        expires_in = float(payload.get("expires_in", 60))
        self._expires_at = time.monotonic() + max(
            0.0, expires_in - _REFRESH_SKEW_SECONDS
        )


def is_opik_configured() -> bool:
    """Return True if all settings required for Opik tracing are present."""
    return all(
        (
            settings.OPIK_URL_OVERRIDE,
            settings.OPIK_KEYCLOAK_TOKEN_URL,
            settings.OPIK_KEYCLOAK_CLIENT_ID,
            settings.OPIK_KEYCLOAK_CLIENT_SECRET,
        )
    )


def configure_opik_keycloak_auth() -> bool:
    """Register the Keycloak auth flow on the Opik SDK's httpx client.

    Must run before the first SDK call (the first ``opik.Opik()``, tracer, or
    ``@opik.track`` function): the hook mutates a process-global list the SDK
    reads when it lazily constructs its ``httpx.Client``.

    Returns True if the hook was registered (or already was), False if Opik
    is not configured.
    """
    global _configured  # noqa: PLW0603

    if not is_opik_configured():
        log.debug("Opik settings not configured, skipping Keycloak auth hook")
        return False
    if _configured:
        return True

    from opik.hooks import HttpxClientHook, add_httpx_client_hook

    # Honor the same TLS verification setting the SDK uses.
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    verify: bool | str = ssl_cert_file if ssl_cert_file else True

    auth = KeycloakClientCredentialsAuth(
        token_url=settings.OPIK_KEYCLOAK_TOKEN_URL,
        client_id=settings.OPIK_KEYCLOAK_CLIENT_ID,
        client_secret=settings.OPIK_KEYCLOAK_CLIENT_SECRET,
        scope=settings.OPIK_KEYCLOAK_SCOPE,
        verify=verify,
    )

    add_httpx_client_hook(
        HttpxClientHook(
            client_modifier=None,
            client_init_arguments={"auth": auth},
        )
    )
    _configured = True
    log.info("Opik Keycloak auth hook registered")
    return True
