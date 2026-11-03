"""Tests for Sentry event scrubbing."""

import json
import logging

import pytest
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import Transport

from main.sentry import (
    before_send,
    scrub_pg_detail,
    scrub_pg_details,
)

# A real MITXONLINE-6PK exception value, with the learner identifiers replaced.
PG_INTEGRITY_ERROR = (
    'null value in column "name" of relation "users_user" violates not-null '
    "constraint\n"
    "DETAIL:  Failing row contains (1863408, , 2026-08-07 18:38:14.503726+00, f, "
    "learner@example.invalid, learner@example.invalid, null, f, t, "
    "12d7dfc5-6f84-46db-9383-2d7079434173, 1863408, learner@example.invalid, f)."
)
PG_PRIMARY_MESSAGE = (
    'null value in column "name" of relation "users_user" violates not-null constraint'
)


class FakeTransport(Transport):
    """Collect outgoing events instead of sending them."""

    def __init__(self):
        super().__init__()
        self.events = []

    def capture_envelope(self, envelope):
        self.events.extend(
            item.payload.json for item in envelope.items if item.type == "event"
        )


@pytest.fixture
def sentry_transport():
    """Initialize the real SDK with before_send, and detach it afterwards."""
    transport = FakeTransport()
    sentry_sdk.init(
        dsn="https://k@o0.ingest.sentry.io/0",
        transport=transport,
        before_send=before_send,
        default_integrations=False,
        integrations=[
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)
        ],
    )
    yield transport
    sentry_sdk.get_global_scope().set_client(None)


def test_detail_line_is_truncated():
    """The row echo goes; the primary error that names the failure stays."""
    scrubbed = scrub_pg_detail(PG_INTEGRITY_ERROR)
    assert scrubbed.startswith(PG_PRIMARY_MESSAGE)
    assert "learner@example.invalid" not in scrubbed
    assert "12d7dfc5-6f84-46db-9383-2d7079434173" not in scrubbed


def test_escaped_detail_line_is_truncated():
    """repr() turns the newline into a literal backslash-n; that form goes too."""
    scrubbed = scrub_pg_detail(repr(Exception(PG_INTEGRITY_ERROR)))
    assert PG_PRIMARY_MESSAGE in scrubbed
    assert "learner@example.invalid" not in scrubbed


def test_message_without_detail_is_unchanged():
    """A message with no DETAIL line passes through untouched."""
    message = "connection to server failed"
    assert scrub_pg_detail(message) == message


def test_hint_and_context_after_detail_are_dropped():
    """HINT and CONTEXT follow DETAIL and can quote row data too."""
    text = "boom\nDETAIL:  row data\nHINT:  try again\nCONTEXT:  SQL statement"
    scrubbed = scrub_pg_detail(text)
    assert "row data" not in scrubbed
    assert "try again" not in scrubbed
    assert "SQL statement" not in scrubbed


def test_scrubs_exception_values_logentry_and_message():
    """Every place the SDK can put an error string is covered."""
    event = {
        "exception": {"values": [{"value": PG_INTEGRITY_ERROR}]},
        "logentry": {
            "message": PG_INTEGRITY_ERROR,
            "formatted": PG_INTEGRITY_ERROR,
        },
        "message": PG_INTEGRITY_ERROR,
    }
    scrub_pg_details(event)
    assert "learner@example.invalid" not in repr(event)


def test_before_send_scrubs_the_event():
    """The scrub is wired into the before_send hook, not just callable."""
    event = {"exception": {"values": [{"value": PG_INTEGRITY_ERROR}]}}
    assert "learner@example.invalid" not in repr(before_send(event, {}))


def test_scrubs_breadcrumb_messages():
    """LoggingIntegration records the log message as a breadcrumb."""
    event = {
        "breadcrumbs": {
            "values": [
                {
                    "type": "log",
                    "category": "django_scim.views",
                    "message": PG_INTEGRITY_ERROR,
                }
            ]
        }
    }
    scrub_pg_details(event)
    assert "learner@example.invalid" not in repr(event)


def test_scrubs_logentry_params():
    """logger.error("...: %s", exc) puts the repr'd exception in logentry.params."""
    event = {
        "logentry": {
            "message": "Unable to complete SCIM call: %s",
            "formatted": "Unable to complete SCIM call: " + PG_INTEGRITY_ERROR,
            "params": [repr(Exception(PG_INTEGRITY_ERROR))],
        }
    }
    scrub_pg_details(event)
    assert "learner@example.invalid" not in repr(event)


def test_scrubs_captured_frame_locals():
    """include_local_variables defaults to True, so repr'd frame vars carry it."""
    event = {
        "exception": {
            "values": [
                {
                    "value": "boom",
                    "stacktrace": {
                        "frames": [
                            {
                                "function": "save",
                                "vars": {
                                    "exc": repr(Exception(PG_INTEGRITY_ERROR)),
                                    "retries": 3,
                                },
                            }
                        ]
                    },
                }
            ]
        }
    }
    scrub_pg_details(event)
    assert "learner@example.invalid" not in repr(event)
    frame = event["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert frame["vars"]["retries"] == 3


def test_walk_preserves_non_string_leaves():
    """The walk must not coerce timestamps, ints or None into strings."""
    event = {
        "timestamp": 1757345533.179,
        "level": "error",
        "extra": {"count": 42, "missing": None, "flag": True},
        "message": PG_INTEGRITY_ERROR,
    }
    scrub_pg_details(event)
    assert event["timestamp"] == 1757345533.179
    assert event["extra"] == {"count": 42, "missing": None, "flag": True}
    assert "learner@example.invalid" not in event["message"]


def test_real_sdk_scrubs_params_and_local_variables(sentry_transport):
    """Go through the real SDK, which repr()s params and locals before before_send."""

    def save():
        exc = Exception(PG_INTEGRITY_ERROR)
        raise exc

    try:
        save()
    except Exception as e:  # noqa: BLE001
        logging.getLogger("x").error("Unable to save: %s", e)  # noqa: TRY400
        sentry_sdk.capture_exception(e)
    sentry_sdk.flush()

    assert len(sentry_transport.events) == 2
    for event in sentry_transport.events:
        assert "learner@example.invalid" not in json.dumps(event)


# A support intake frame: the reply-to address and the learner's own words are
# arguments to the call that files the ticket.
CHATBOT_FRAME_VARS = {
    "email": "'learner@example.invalid'",
    "description": "'I cannot log in to the week 3 video'",
}


def _frames_event(frames):
    return {
        "exception": {"values": [{"value": "boom", "stacktrace": {"frames": frames}}]}
    }


def test_drops_locals_from_chatbot_frames():
    """
    No DETAIL line to truncate here: the learner's address is a plain argument,
    and the SDK ships frame locals by default.
    """
    event = _frames_event(
        [
            {
                "module": "ai_chatbots.zendesk",
                "function": "file_support_ticket",
                "filename": "ai_chatbots/zendesk.py",
                "lineno": 193,
                "vars": dict(CHATBOT_FRAME_VARS),
            },
            {"module": "main.views", "function": "index", "vars": {"retries": 3}},
        ]
    )

    before_send(event, {})

    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    assert "learner@example.invalid" not in json.dumps(event)
    # What identifies the failure stays; only the learner's data goes.
    assert frames[0]["function"] == "file_support_ticket"
    assert frames[0]["lineno"] == 193
    # The frames around it go too: the learner's data is passed through them.
    assert "vars" not in frames[1]


def test_drops_locals_from_library_frames_in_a_chatbot_event():
    """
    The serializer's repr renders the whole request body, and the frame holding
    it belongs to rest_framework. Scoping the scrub to the chatbot package left
    the learner's message, address and page url in the event.
    """
    event = _frames_event(
        [
            {"module": "ai_chatbots.consumers", "function": "handle", "vars": {}},
            {
                "module": "rest_framework.serializers",
                "function": "is_valid",
                "vars": {
                    "self": (
                        "SupportChatRequestSerializer(data={'message': 'I cannot "
                        "log in', 'user_email': 'learner@example.invalid'}):"
                    )
                },
            },
        ]
    )

    before_send(event, {})

    assert "learner@example.invalid" not in json.dumps(event)


def test_keeps_locals_when_no_chatbot_frame_is_present():
    """
    Every other app keeps the locals that make a traceback readable, which is
    why this is not just include_local_variables turned off.
    """
    event = _frames_event(
        [{"module": "main.views", "function": "index", "vars": {"retries": 3}}]
    )

    before_send(event, {})

    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    assert frames[0]["vars"] == {"retries": 3}


def test_drops_locals_from_chatbot_frames_in_thread_stacktraces():
    """Stacktraces reach Sentry under threads as well as exception."""
    event = {
        "threads": {
            "values": [
                {
                    "stacktrace": {
                        "frames": [
                            {
                                "module": "ai_chatbots.consumers",
                                "function": "handle",
                                "vars": dict(CHATBOT_FRAME_VARS),
                            }
                        ]
                    }
                }
            ]
        }
    }

    before_send(event, {})

    assert "learner@example.invalid" not in json.dumps(event)


def test_keeps_locals_from_a_module_that_merely_starts_the_same():
    """Prefix matching on the package name, not on the string."""
    event = _frames_event(
        [{"module": "ai_chatbots_extra.thing", "vars": {"email": "a@b.invalid"}}]
    )

    before_send(event, {})

    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    assert frames[0]["vars"] == {"email": "a@b.invalid"}


def test_real_sdk_drops_chatbot_frame_locals(sentry_transport):
    """
    Go through the real SDK, which fills frame vars from the live frames. The
    raiser is exec'd with a chatbot __name__ because that is what the SDK reads
    the frame's module out of.

    Asserted on the frame rather than the whole event: the address is a literal
    in this file, so the SDK's source context carries it no matter what the
    scrub does. Real learner data never arrives as a source line.
    """
    namespace = {"__name__": "ai_chatbots.support_probe"}
    exec(  # noqa: S102
        "def raiser(email):\n    raise RuntimeError('kaboom')", namespace
    )

    try:
        namespace["raiser"]("learner@example.invalid")
    except Exception as e:  # noqa: BLE001
        sentry_sdk.capture_exception(e)
    sentry_sdk.flush()

    frames = sentry_transport.events[0]["exception"]["values"][0]["stacktrace"][
        "frames"
    ]
    chatbot_frames = [
        frame for frame in frames if frame.get("module") == "ai_chatbots.support_probe"
    ]
    assert chatbot_frames, [frame.get("module") for frame in frames]
    assert "vars" not in chatbot_frames[0]
