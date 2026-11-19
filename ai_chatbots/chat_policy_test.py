"""Stream behaviour of ChatLiteLLM under ol-llm's call policy, against a local server.

The server speaks the OpenAI chat completions stream, so these run litellm and the
OpenAI SDK as production does and check what the policy promises: the timeout
bounds the gap between chunks, not the whole reply, and nothing is retried once a
stream has started.
"""

import asyncio
import json
import time

import httpx
import litellm
import ol_llm
import pytest
from aiohttp import web
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from litellm.exceptions import MidStreamFallbackError

from ai_chatbots.conftest import chat_completions_server

USE_CASE = ol_llm.UseCase(
    app="learn-ai", name="chat-policy-test", profile="request", model="openai:gpt-4o"
)
TIMEOUT_SECONDS = 1.0
POLICY_ENVIRON = {
    "OL_LLM_CHAT_POLICY_TEST_TIMEOUT_SECONDS": str(TIMEOUT_SECONDS),
    "OL_LLM_CHAT_POLICY_TEST_MAX_RETRIES": "2",
}


def _chunk(content: str) -> bytes:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body)}\n\n".encode()


class FakeProvider:
    """An OpenAI-compatible server that follows a script per request.

    :param scripts: one entry per request in arrival order: ``("status", code)``
        or ``("status", code, headers)``, or ``("stream", delays, ending)`` to
        stream one chunk after each delay, then ``"abort"`` to drop the connection
        instead of finishing, or ``("error_event", error)`` to send an error object
        inside a 200 stream, as a proxy reports an upstream failure.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = 0
        self.retry_counts: list[str | None] = []
        self.api_base = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        await request.read()
        # The OpenAI SDK numbers its own retries in this header.
        self.retry_counts.append(request.headers.get("x-stainless-retry-count"))
        script = self.scripts[min(self.requests, len(self.scripts) - 1)]
        self.requests += 1
        kind, *args = script
        if kind == "status":
            status, *headers = args
            return web.json_response(
                {"error": {"message": "failed", "type": "server_error"}},
                status=status,
                headers=headers[0] if headers else None,
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if kind == "error_event":
            await response.write(f"data: {json.dumps({'error': args[0]})}\n\n".encode())
            await response.write_eof()
            return response
        delays, ending = args
        for index, delay in enumerate(delays):
            await asyncio.sleep(delay)
            await response.write(_chunk(f"part{index} "))
        if ending == "abort":
            request.transport.close()
            return response
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


@pytest.fixture
async def provider():
    """Serve a FakeProvider on a free local port; tests set its scripts."""
    fake = FakeProvider([])
    async with chat_completions_server(fake.handle) as api_base:
        fake.api_base = api_base
        yield fake


def _chat_model(api_base: str) -> ol_llm.PolicyChatModel:
    settings = ol_llm.chat_litellm_kwargs(USE_CASE, environ=POLICY_ENVIRON)
    llm = ChatLiteLLM(
        model="openai/gpt-4o",
        api_base=api_base,
        api_key="test-key",
        streaming=True,
        **settings,
    )
    return ol_llm.with_call_policy(llm, USE_CASE, environ=POLICY_ENVIRON)


def _caused_by(error: BaseException, error_type: type[BaseException]) -> bool:
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, error_type):
            return True
        seen = (
            getattr(seen, "original_exception", None)
            or seen.__cause__
            or seen.__context__
        )
    return False


async def _collect(llm: ol_llm.PolicyChatModel, received: list[str]) -> None:
    # A loop, not extend(), so the chunks before a failure stay recorded.
    async for chunk in llm.astream([HumanMessage("hello")]):
        if chunk.content:
            received.append(str(chunk.content))  # noqa: PERF401


async def test_a_stream_longer_than_the_timeout_completes(provider):
    """Chunks that each arrive within the timeout finish a reply that outlasts it."""
    provider.scripts = [("stream", [0.4] * 6, "finish")]
    received: list[str] = []

    await _collect(_chat_model(provider.api_base), received)

    assert len(received) == 6
    assert provider.requests == 1


async def test_a_gap_longer_than_the_timeout_fails_without_a_retry(provider):
    """A stream that stalls after its first chunk times out and is not replayed."""
    stall_seconds = TIMEOUT_SECONDS * 3
    provider.scripts = [("stream", [0, stall_seconds], "finish")]
    received: list[str] = []
    started = time.monotonic()

    # litellm's stream handler reports a read timeout as a connection error.
    with pytest.raises(MidStreamFallbackError) as raised:
        await _collect(_chat_model(provider.api_base), received)

    assert time.monotonic() - started < stall_seconds
    assert _caused_by(raised.value, httpx.ReadTimeout)
    assert received == ["part0 "]
    assert provider.requests == 1


async def test_a_dropped_stream_fails_without_a_retry(provider):
    """A connection lost mid-stream reaches the caller and is not replayed."""
    provider.scripts = [("stream", [0, 0.1], "abort")]
    received: list[str] = []

    with pytest.raises(MidStreamFallbackError) as raised:
        await _collect(_chat_model(provider.api_base), received)

    assert isinstance(raised.value.original_exception, litellm.APIConnectionError)
    assert received == ["part0 ", "part1 "]
    assert provider.requests == 1


class _RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        self.events: list[str] = []

    def on_chat_model_start(self, *args, **kwargs):  # noqa: ARG002
        self.events.append("start")

    def on_llm_new_token(self, *args, **kwargs):  # noqa: ARG002
        self.events.append("token")

    def on_llm_end(self, *args, **kwargs):  # noqa: ARG002
        self.events.append("end")


async def test_callbacks_assigned_after_construction_reach_the_call(provider):
    """TutorBot sets callbacks on the wrapper, and message_tutor binds tools to it."""
    provider.scripts = [("stream", [0, 0], "finish")]
    llm = _chat_model(provider.api_base)
    handler = _RecordingHandler()
    llm.callbacks = [handler]
    bound = llm.bind_tools(
        [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    )

    await asyncio.to_thread(bound.invoke, [HumanMessage("hello")])

    # The stream's empty and finish chunks are tokens too, so only the shape is fixed.
    assert handler.events[0] == "start"
    assert handler.events[-1] == "end"
    assert set(handler.events[1:-1]) == {"token"}


async def test_a_failure_before_the_stream_starts_is_retried(provider):
    """A 5xx before any chunk is retried by ol-llm, not by the OpenAI SDK."""
    provider.scripts = [("status", 500), ("status", 503), ("stream", [0], "finish")]
    received: list[str] = []

    await _collect(_chat_model(provider.api_base), received)

    assert received == ["part0 "]
    # Each request is a first SDK attempt: the SDK's own retries are off.
    assert provider.retry_counts == ["0", "0", "0"]


async def test_a_stall_before_the_first_chunk_is_retried(provider):
    """A stream that sends headers and then nothing is retried, having shown nothing."""
    provider.scripts = [
        ("stream", [TIMEOUT_SECONDS * 3], "finish"),
        ("stream", [0], "finish"),
    ]
    received: list[str] = []

    await _collect(_chat_model(provider.api_base), received)

    assert received == ["part0 "]
    assert provider.requests == 2


async def test_retries_stop_at_the_policy(provider):
    """Past the policy's retries the failure reaches the caller."""
    provider.scripts = [("status", 500)]
    received: list[str] = []

    with pytest.raises(litellm.InternalServerError):
        await _collect(_chat_model(provider.api_base), received)

    assert provider.requests == 3


async def test_a_retry_after_past_the_budget_ends_the_call(provider):
    """A 429 asking for longer than the request budget fails at once."""
    provider.scripts = [
        ("status", 429, {"Retry-After": "30"}),
        ("stream", [0], "finish"),
    ]
    started = time.monotonic()

    with pytest.raises(litellm.RateLimitError):
        await _collect(_chat_model(provider.api_base), [])

    assert time.monotonic() - started < 5
    assert provider.requests == 1


async def test_a_retry_after_within_the_budget_is_waited_out(provider):
    """A 429 asking for less than the budget is retried after that wait."""
    provider.scripts = [
        ("status", 429, {"Retry-After": "1"}),
        ("stream", [0], "finish"),
    ]
    received: list[str] = []
    started = time.monotonic()

    await _collect(_chat_model(provider.api_base), received)

    assert time.monotonic() - started >= 1
    assert received == ["part0 "]
    assert provider.requests == 2


async def test_an_error_object_in_the_stream_is_not_retried(provider):
    """An error the provider sends inside a 200 stream is a rejection, not an outage."""
    provider.scripts = [
        (
            "error_event",
            {
                "message": "too long",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            },
        ),
        ("stream", [0], "finish"),
    ]

    with pytest.raises(MidStreamFallbackError):
        await _collect(_chat_model(provider.api_base), [])

    assert provider.requests == 1
