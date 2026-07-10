"""Tests for Keycloak client-credentials auth for the Opik SDK."""

import json

import httpx
import pytest

from main import opik_keycloak_auth
from main.opik_keycloak_auth import (
    KeycloakClientCredentialsAuth,
    configure_opik_keycloak_auth,
    is_opik_configured,
)

TOKEN_URL = "https://sso.example.edu/realms/olapps/protocol/openid-connect/token"  # noqa: S105


@pytest.fixture
def opik_settings(settings):
    """Set all required Opik settings."""
    settings.OPIK_URL_OVERRIDE = "https://opik-ci.example.edu/api/"
    settings.OPIK_KEYCLOAK_TOKEN_URL = TOKEN_URL
    settings.OPIK_KEYCLOAK_CLIENT_ID = "ol-opik-client"
    settings.OPIK_KEYCLOAK_CLIENT_SECRET = "secret"  # noqa: S105
    settings.OPIK_KEYCLOAK_SCOPE = None
    return settings


def make_auth(token_responses):
    """Build an auth instance whose token client replays canned responses."""
    responses = iter(token_responses)

    def token_endpoint(request):
        assert request.url == httpx.URL(TOKEN_URL)
        body = dict(
            pair.split("=") for pair in request.content.decode().split("&") if pair
        )
        assert body["grant_type"] == "client_credentials"
        return next(responses)

    auth = KeycloakClientCredentialsAuth(
        token_url=TOKEN_URL,
        client_id="ol-opik-client",
        client_secret="secret",  # noqa: S106
    )
    auth._token_client = httpx.Client(  # noqa: SLF001
        transport=httpx.MockTransport(token_endpoint)
    )
    return auth


def token_response(token, expires_in=300):
    """Return a canned Keycloak token endpoint response."""
    return httpx.Response(200, json={"access_token": token, "expires_in": expires_in})


def test_auth_fetches_and_caches_token():
    """The first request fetches a token; later requests reuse the cache."""
    auth = make_auth([token_response("token-1")])
    seen_headers = []

    def api(request):
        seen_headers.append(request.headers["Authorization"])
        return httpx.Response(200, json={})

    client = httpx.Client(auth=auth, transport=httpx.MockTransport(api))
    client.get("https://opik.example.edu/api/foo")
    client.get("https://opik.example.edu/api/bar")

    assert seen_headers == ["Bearer token-1", "Bearer token-1"]


def test_auth_refreshes_expired_token():
    """An expired cached token is replaced before the request is sent."""
    auth = make_auth([token_response("token-1"), token_response("token-2")])

    def api(request):
        return httpx.Response(200, json={})

    client = httpx.Client(auth=auth, transport=httpx.MockTransport(api))
    client.get("https://opik.example.edu/api/foo")
    auth._expires_at = 0.0  # noqa: SLF001 - simulate expiry
    request = client.build_request("GET", "https://opik.example.edu/api/bar")
    sent = client.send(request)

    assert sent.request.headers["Authorization"] == "Bearer token-2"


def test_auth_retries_once_on_401():
    """A 401 triggers a forced token refresh and a single retry."""
    auth = make_auth([token_response("stale"), token_response("fresh")])
    api_calls = []

    def api(request):
        api_calls.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer stale":
            return httpx.Response(401)
        return httpx.Response(200, json={})

    client = httpx.Client(auth=auth, transport=httpx.MockTransport(api))
    response = client.get("https://opik.example.edu/api/foo")

    assert response.status_code == 200
    assert api_calls == ["Bearer stale", "Bearer fresh"]


def test_auth_scope_included_when_set():
    """The optional scope is sent to the token endpoint."""

    def token_endpoint(request):
        body = dict(
            pair.split("=") for pair in request.content.decode().split("&") if pair
        )
        assert body["scope"] == "opik"
        return token_response("token-1")

    auth = KeycloakClientCredentialsAuth(
        token_url=TOKEN_URL,
        client_id="ol-opik-client",
        client_secret="secret",  # noqa: S106
        scope="opik",
    )
    auth._token_client = httpx.Client(  # noqa: SLF001
        transport=httpx.MockTransport(token_endpoint)
    )
    assert auth._get_token() == "token-1"  # noqa: SLF001


def test_is_opik_configured(settings):
    """is_opik_configured requires all of URL, token URL, client id and secret."""
    settings.OPIK_URL_OVERRIDE = None
    settings.OPIK_KEYCLOAK_TOKEN_URL = None
    settings.OPIK_KEYCLOAK_CLIENT_SECRET = None
    assert is_opik_configured() is False


def test_is_opik_configured_true(opik_settings):
    """is_opik_configured is True when everything is set."""
    assert is_opik_configured() is True


def test_configure_skips_when_unconfigured(settings, mocker):
    """No hook is registered when Opik settings are absent."""
    settings.OPIK_URL_OVERRIDE = None
    settings.OPIK_KEYCLOAK_TOKEN_URL = None
    settings.OPIK_KEYCLOAK_CLIENT_SECRET = None
    mocker.patch.object(opik_keycloak_auth, "_configured", new=False)
    add_hook = mocker.patch("opik.hooks.add_httpx_client_hook")

    assert configure_opik_keycloak_auth() is False
    add_hook.assert_not_called()


def test_configure_registers_hook_once(opik_settings, mocker):
    """The httpx client hook is registered exactly once with our auth."""
    mocker.patch.object(opik_keycloak_auth, "_configured", new=False)
    add_hook = mocker.patch("opik.hooks.add_httpx_client_hook")

    assert configure_opik_keycloak_auth() is True
    assert configure_opik_keycloak_auth() is True

    add_hook.assert_called_once()
    hook = add_hook.call_args[0][0]
    auth = hook._httpx_client_arguments["auth"]  # noqa: SLF001
    assert isinstance(auth, KeycloakClientCredentialsAuth)
    assert auth._token_url == TOKEN_URL  # noqa: SLF001


def test_token_endpoint_error_propagates():
    """A failing token endpoint raises instead of caching a bad token."""
    auth = make_auth([httpx.Response(500, text=json.dumps({"error": "boom"}))])
    with pytest.raises(httpx.HTTPStatusError):
        auth._get_token()  # noqa: SLF001
