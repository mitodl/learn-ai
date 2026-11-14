"""Tests for consumer throttle classes."""

from importlib import import_module
from pathlib import Path
from time import time

import pytest
from asgiref.sync import sync_to_async
from django.conf import settings as django_settings
from django.contrib.auth.models import AnonymousUser
from django.core.cache import caches
from freezegun import freeze_time
from rest_framework import exceptions

import main
from ai_chatbots.consumers import BaseBotHttpConsumer
from main.constants import DURATION_MAPPING
from main.consumer_throttles import CONSUMER_THROTTLES_KEY, UserScopedRateThrottle
from main.consumers import BaseThrottledAsyncConsumer
from main.factories import ConsumerThrottleLimitFactory


@pytest.fixture(autouse=True)
def shared_throttle_cache(settings):
    """
    Give the redis alias a working backend; the root conftest makes it a
    DummyCache, which would forget every request the throttle records.
    """
    settings.CELERY_BROKER_URL = "redis://redis:6379/4"
    settings.CACHES = {
        **settings.CACHES,
        "redis": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "throttle-test-redis",
        },
    }
    yield caches["redis"]
    caches["redis"].clear()


@pytest.mark.parametrize(
    ("is_authenticated", "is_staff"), [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("interval", ["minute", "hour", "day", "week"])
async def test_scoped_throttle_parse_rate(user, is_authenticated, interval, is_staff):
    """Test the parse_rate method of ScopedRateThrottle."""
    throttle_limit = await sync_to_async(ConsumerThrottleLimitFactory.create)(
        interval=interval
    )
    user = user if is_authenticated else AnonymousUser()
    if is_authenticated:
        user.is_staff = is_staff
    rate = await UserScopedRateThrottle().parse_rate(throttle_limit.__dict__, user)
    assert (
        rate == (0, 0)
        if is_staff
        else (
            throttle_limit.auth_limit
            if is_authenticated
            else throttle_limit.anon_limit,
            DURATION_MAPPING[interval],
        )
    )


@pytest.mark.parametrize("has_consumer_scope", [True, False])
@pytest.mark.parametrize("over_limit", [True, False])
async def test_scoped_allow_request(user, has_consumer_scope, over_limit):
    """Test the allow_request method of ScopedRateThrottle."""
    throttle_limit = await sync_to_async(ConsumerThrottleLimitFactory.create)()
    caches["default"].set(CONSUMER_THROTTLES_KEY, None)
    consumer = BaseThrottledAsyncConsumer()
    consumer.scope = {"user": user} if has_consumer_scope else {}
    consumer.throttle_scope = throttle_limit.throttle_key
    consumer.throttle_classes = [UserScopedRateThrottle]
    caches["redis"].set(
        f"throttle_{throttle_limit.throttle_key}_{user.global_id}",
        [time() for t in range(10 if over_limit else 1)],
    )
    rate = UserScopedRateThrottle()
    allowed = await rate.allow_request(consumer)
    if not has_consumer_scope:
        assert allowed is True
    else:
        assert allowed is not over_limit


@pytest.mark.parametrize(
    ("now", "history", "duration", "num_requests", "expected"),
    [
        (100, [50], 100, 10, 5.0),
        (80, [50], 200, 10, 17.0),
        (80, [50], 200, 1, 170.0),
        (75, [50], 200, 0, None),
        (100, [50], 25, 1, -25.0),
        (100, [50], 3600, 1, 3550.0),
    ],
)
async def test_wait(now, history, duration, num_requests, expected):
    rate = UserScopedRateThrottle()
    rate.now = now
    rate.history = history
    rate.duration = duration
    rate.num_requests = num_requests
    assert await rate.wait() == expected


def test_every_bot_consumer_has_a_seeded_throttle_limit():
    """
    A consumer whose throttle_scope has no ConsumerThrottleLimit row raises
    ImproperlyConfigured on the first live request, so every bot needs a data
    migration seeding its row.
    """

    def bot_consumers(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from bot_consumers(subclass)

    scopes = {
        consumer.throttle_scope
        for consumer in bot_consumers(BaseBotHttpConsumer)
        if getattr(consumer, "throttle_scope", None)
    }
    # Read the migrations rather than the database: a transactional test flushes
    # the rows data migrations seeded, and --reuse-db keeps the empty table.
    migrations_dir = Path(main.__file__).parent / "migrations"
    sources = "\n".join(
        path.read_text() for path in sorted(migrations_dir.glob("0*.py"))
    )

    assert {scope for scope in scopes if f'"{scope}"' not in sources} == set()


def _anon_consumer(throttle_key, *, forwarded_for, client_ip="10.0.0.2"):
    """
    Build an anonymous consumer the way a live request arrives: through APISIX,
    with a fresh, empty signed-cookie session (no sessionid ever reaches the
    browser on these routes), not conftest's stable mock session.
    """
    session = import_module(django_settings.SESSION_ENGINE).SessionStore()
    consumer = BaseThrottledAsyncConsumer()
    consumer.scope = {
        "user": AnonymousUser(),
        "session": session,
        "client": (client_ip, 54321),
        "headers": [(b"x-forwarded-for", forwarded_for.encode())],
    }
    consumer.throttle_scope = throttle_key
    return consumer


@pytest.fixture
async def anon_limit_two(db, settings):
    """Return a day-long throttle row allowing two anonymous requests."""
    settings.REST_FRAMEWORK = {**settings.REST_FRAMEWORK, "NUM_PROXIES": 1}
    # Created through sync_to_async so the throttle's own sync_to_async read
    # runs on the same DB connection and sees the row.
    return await sync_to_async(ConsumerThrottleLimitFactory.create)(
        anon_limit=2, interval="day"
    )


@pytest.fixture
def rotating_session_clock():
    """
    Advance the clock past a second on every read, so each request's empty
    signed-cookie session gets a new key, as it does in production.
    """
    with freeze_time(auto_tick_seconds=1.1):
        yield


async def test_anonymous_throttle_holds_across_rotating_sessions(
    anon_limit_two, rotating_session_clock
):
    """The session key changes every second; the throttle must not."""
    consumers = [
        _anon_consumer(anon_limit_two.throttle_key, forwarded_for="203.0.113.5")
        for _ in range(3)
    ]
    for consumer in consumers[:2]:
        await consumer.check_throttles()

    with pytest.raises(exceptions.Throttled):
        await consumers[2].check_throttles()
    assert len({consumer.get_session_key() for consumer in consumers}) == 3


async def test_anonymous_throttle_is_per_address(
    anon_limit_two, rotating_session_clock
):
    """A caller at another address is not blocked by someone else's usage."""
    for _ in range(2):
        await _anon_consumer(
            anon_limit_two.throttle_key, forwarded_for="203.0.113.5"
        ).check_throttles()
    with pytest.raises(exceptions.Throttled):
        await _anon_consumer(
            anon_limit_two.throttle_key, forwarded_for="203.0.113.5"
        ).check_throttles()

    await _anon_consumer(
        anon_limit_two.throttle_key, forwarded_for="198.51.100.7"
    ).check_throttles()


async def test_anonymous_throttle_ignores_spoofed_forwarded_for(
    anon_limit_two, rotating_session_clock
):
    """A client-supplied X-Forwarded-For entry does not buy a fresh bucket."""
    for spoof in ("1.1.1.1", "2.2.2.2"):
        await _anon_consumer(
            anon_limit_two.throttle_key, forwarded_for=f"{spoof}, 203.0.113.5"
        ).check_throttles()

    with pytest.raises(exceptions.Throttled):
        await _anon_consumer(
            anon_limit_two.throttle_key, forwarded_for="3.3.3.3, 203.0.113.5"
        ).check_throttles()


async def test_anonymous_throttle_counts_requests_other_pods_recorded(
    anon_limit_two, shared_throttle_cache
):
    """
    Each pod has its own default cache, so the history has to live in the
    shared one for the limit to hold across replicas.
    """
    shared_throttle_cache.set(
        f"throttle_{anon_limit_two.throttle_key}_203.0.113.5", [time(), time()]
    )

    with pytest.raises(exceptions.Throttled):
        await _anon_consumer(
            anon_limit_two.throttle_key, forwarded_for="203.0.113.5"
        ).check_throttles()


def test_throttle_history_uses_default_cache_without_redis(settings):
    """With no broker configured, the redis alias has nowhere to connect to."""
    settings.CELERY_BROKER_URL = None

    assert UserScopedRateThrottle().cache is caches["default"]
