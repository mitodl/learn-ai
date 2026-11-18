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
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from litellm.exceptions import MidStreamFallbackError

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

    :param scripts: one entry per request in arrival order: ``("status", code)``,
        or ``("stream", delays)`` to stream one chunk after each delay, then
        ``"abort"`` to drop the connection instead of finishing.
    """

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = 0
        self.api_base = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        await request.read()
        script = self.scripts[min(self.requests, len(self.scripts) - 1)]
        self.requests += 1
        kind, *args = script
        if kind == "status":
            return web.json_response(
                {"error": {"message": "failed", "type": "server_error"}},
                status=args[0],
            )
        delays, ending = args
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
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
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    fake.api_base = f"http://127.0.0.1:{port}/v1"
    yield fake
    await runner.cleanup()


def _chat_model(api_base: str) -> ChatLiteLLM:
    policy = ol_llm.chat_litellm_kwargs(USE_CASE, environ=POLICY_ENVIRON)
    return ChatLiteLLM(
        model="openai/gpt-4o",
        api_base=api_base,
        api_key="test-key",
        streaming=True,
        **policy,
    )


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


async def _collect(llm: ChatLiteLLM, received: list[str]) -> None:
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


async def test_a_failure_before_the_stream_starts_is_retried(provider):
    """A 5xx before any chunk is retried up to the policy's retries."""
    provider.scripts = [("status", 500), ("status", 503), ("stream", [0], "finish")]
    received: list[str] = []

    await _collect(_chat_model(provider.api_base), received)

    assert received == ["part0 "]
    assert provider.requests == 3


async def test_retries_stop_at_the_policy(provider):
    """Past the policy's retries the failure reaches the caller."""
    provider.scripts = [("status", 500)]
    received: list[str] = []

    with pytest.raises(litellm.InternalServerError):
        await _collect(_chat_model(provider.api_base), received)

    assert provider.requests == 3
