"""
Read a signed-in learner's own records from MIT Learn, MITx Online and Open edX.

These APIs already return the signed-in learner's own enrollments, grades,
certificates and dates. learn-ai calls them with the learner's own session
cookies, taken from the chat request, so it never holds a credential that can
read anyone else's records.

The MIT Learn cookie is the learner's whole login and stays valid for up to 14
days, so:

* The cookies live only in a context variable for one chat request
  (``forwarded_learner_session``). Never put them in LangGraph state or
  ``config`` (both are saved with checkpoints), tool arguments, model input,
  logs or traces. ``LearnerSession`` never shows their values.
* Each cookie goes only to the host it belongs to, on fixed paths.
* Redirects are not followed, and each call uses its own HTTP client, so any
  cookies an API sets are discarded with it.
"""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import httpx
from django.conf import settings

log = logging.getLogger(__name__)

LEARN = "learn"
OPENEDX = "openedx"

# Host -> (base URL setting, session cookie name setting)
_HOST_SETTINGS = {
    LEARN: ("LEARNER_API_LEARN_BASE_URL", "LEARNER_API_LEARN_COOKIE_NAME"),
    OPENEDX: ("LEARNER_API_OPENEDX_BASE_URL", "LEARNER_API_OPENEDX_COOKIE_NAME"),
}

# Open edX course keys: course-v1:<org>+<course>+<run>
COURSE_KEY_RE = re.compile(r"course-v1:[\w~.:-]+\+[\w~.:-]+\+[\w~.:-]+", re.ASCII)

# RFC 6265 cookie-octet: anything else can't go in a Cookie header as-is
COOKIE_VALUE_RE = re.compile(r"[\x21\x23-\x2B\x2D-\x3A\x3C-\x5B\x5D-\x7E]+")


class LearnerAPIError(Exception):
    """A learner record could not be read. Report it as "couldn't check"."""

    def __init__(self, operation: str, status_code: int | None = None):
        """Record which operation failed and the HTTP status, if there was one."""
        self.operation = operation
        self.status_code = status_code
        detail = f" with status {status_code}" if status_code else ""
        super().__init__(f"{operation} failed{detail}")


class LearnerSession:
    """A signed-in learner's session cookies, per host. Never shows their values."""

    __slots__ = ("_cookies",)

    def __init__(self, cookies: dict[str, dict[str, str]]):
        """Take the cookies to send to each host."""
        self._cookies = cookies

    def has_cookies(self, host: str) -> bool:
        """Whether there are session cookies for this host."""
        return bool(self._cookies.get(host))

    def cookie_header(self, host: str) -> str:
        """Build the Cookie header value for this host."""
        return "; ".join(
            f"{name}={value}" for name, value in self._cookies[host].items()
        )

    def __repr__(self) -> str:
        """Show which hosts have cookies, never the values."""
        hosts = ", ".join(
            f"{host}=<redacted>" for host in self._cookies if self.has_cookies(host)
        )
        return f"LearnerSession({hosts})"

    __str__ = __repr__

    def __reduce__(self):
        """Refuse pickling and copying, so the cookies can't be saved."""
        msg = "LearnerSession cannot be pickled or copied"
        raise TypeError(msg)


_session: ContextVar[LearnerSession | None] = ContextVar(
    "learner_session", default=None
)


def _session_cookies(cookies: dict[str, str], name: str) -> dict[str, str]:
    """
    Get the named session cookie, plus the chunks APISIX splits a large session
    into (``<name>2``, ``<name>3``, ...; see lua-resty-session). A malformed
    value means no session.
    """
    if not name or not cookies.get(name):
        return {}
    found = {name: cookies[name]}
    chunk = 2
    while cookies.get(f"{name}{chunk}"):
        found[f"{name}{chunk}"] = cookies[f"{name}{chunk}"]
        chunk += 1
    if not all(COOKIE_VALUE_RE.fullmatch(value) for value in found.values()):
        return {}
    return found


@contextmanager
def forwarded_learner_session(scope: dict) -> Iterator[None]:
    """
    Make a signed-in learner's session cookies available to this module for the
    duration of one request. Anonymous requests get no session.

    Args:
        scope: the Channels connection scope, with ``user`` and ``cookies``
    """
    user = scope.get("user")
    session = None
    if user is not None and user.is_authenticated:
        request_cookies = scope.get("cookies") or {}
        cookies = {
            host: _session_cookies(request_cookies, getattr(settings, cookie_setting))
            for host, (_, cookie_setting) in _HOST_SETTINGS.items()
        }
        if any(cookies.values()):
            session = LearnerSession(cookies)
    token = _session.set(session)
    try:
        yield
    finally:
        _session.reset(token)


def has_learner_session() -> bool:
    """Whether the current request carries the learner's MIT Learn session."""
    session = _session.get()
    return session is not None and session.has_cookies(LEARN)


async def _get(operation: str, host: str, path: str):
    """
    GET a fixed path on one of the allowed hosts as the current learner, and
    return the decoded JSON.

    Raises:
        LearnerAPIError: with no session for that host, or on any failure
    """
    session = _session.get()
    if session is None or not session.has_cookies(host):
        raise LearnerAPIError(operation)

    base_url = getattr(settings, _HOST_SETTINGS[host][0])
    try:
        base = httpx.URL(base_url)
        url = base.join(path)
    except httpx.InvalidURL:
        url = None
    if (
        url is None
        or base.scheme != "https"
        or not base.host
        or (url.scheme, url.netloc) != (base.scheme, base.netloc)
    ):
        log.error("Learner API %s: base URL %r is not usable", operation, base_url)
        raise LearnerAPIError(operation)

    # Any exception from here is re-raised "from None": the frames and httpx
    # errors under it hold the Cookie header, and must not be chained onto what
    # callers log or send to Sentry.
    try:
        async with httpx.AsyncClient(
            timeout=settings.REQUESTS_TIMEOUT, follow_redirects=False
        ) as client:
            response = await client.get(
                url, headers={"Cookie": session.cookie_header(host)}
            )
    except Exception as error:  # noqa: BLE001
        log.warning("Learner API %s failed: %s", operation, type(error).__name__)
        raise LearnerAPIError(operation) from None

    if not response.is_success:
        log.warning(
            "Learner API %s returned status %s", operation, response.status_code
        )
        raise LearnerAPIError(operation, response.status_code)
    try:
        return response.json()
    except ValueError:
        log.warning("Learner API %s returned invalid JSON", operation)
        raise LearnerAPIError(operation, response.status_code) from None


async def get_enrollments() -> list[dict]:
    """Get the learner's MITx Online course run enrollments, grades, certificates."""
    return await _get("get_enrollments", LEARN, "mitxonline/api/v3/enrollments/")


async def get_program_enrollments() -> list[dict]:
    """Get the learner's MITx Online program enrollments."""
    return await _get(
        "get_program_enrollments", LEARN, "mitxonline/api/v3/program_enrollments/"
    )


async def get_program_certificates() -> list[dict]:
    """Get the learner's MIT Learn program certificates."""
    return await _get("get_program_certificates", LEARN, "api/v0/program_certificates/")


async def get_course_dates(course_key: str) -> dict:
    """Get the learner's dates for one Open edX course run, from its Dates tab."""
    operation = "get_course_dates"
    if not COURSE_KEY_RE.fullmatch(course_key):
        raise LearnerAPIError(operation)
    return await _get(operation, OPENEDX, f"api/course_home/v1/dates/{course_key}")
