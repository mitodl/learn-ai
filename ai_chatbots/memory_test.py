"""Tests for the learner's notes in shared memory"""

import asyncio
import json
import logging

import pytest
from aiohttp import web
from asgiref.sync import sync_to_async
from ol_llm.memory import InMemoryBackend, InMemoryStore, Memory, MemoryFenced
from redis.exceptions import LockNotOwnedError

from ai_chatbots import memory
from ai_chatbots.chatbots import MemoryExtractionBot, MemoryGateBot
from ai_chatbots.memory import (
    ABOUT_KEY,
    INSTRUCTIONS_KEY,
    TUTOR_RULES_WIN,
    GateDecision,
    MemoryRevision,
    fit_length,
    open_learner_notes,
    session_token,
)

pytestmark = pytest.mark.django_db

TOKEN = "learner-session-token"  # noqa: S105 - the in-memory backend's learner id


@pytest.fixture(autouse=True)
def redis_lock(mocker):
    """Stand in for the per-learner Redis lock (tests run on a dummy cache)."""
    lock = mocker.Mock()
    lock.acquire.return_value = True
    mocker.patch(
        "ai_chatbots.memory.get_redis_connection"
    ).return_value.lock.return_value = lock
    return lock


@pytest.fixture
def store():
    """Return one shared memory store, as the deployed service is."""
    return InMemoryStore()


@pytest.fixture
def learn_ai_memory(mocker, settings, store):
    """Turn memory on for learn-ai against the in-memory backend."""
    settings.OL_LLM_MEMORY_BACKEND = "in-memory"
    learn_ai = Memory(InMemoryBackend("learn-ai", store=store))
    mocker.patch("ai_chatbots.memory.get_memory", return_value=learn_ai)
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=True)
    return learn_ai


@pytest.fixture
async def seeded(learn_ai_memory, store):
    """Seed mit-learn's profile notes and learn-ai's own notes for the learner."""
    profile = await Memory(InMemoryBackend("mit-learn", store=store)).learner(
        TOKEN, use_case="test"
    )
    await profile.put("profile-current-education", "Education level: Bachelor's.")
    await profile.put("profile-topic-interests", "Interested in: Data Science.")
    await profile.put("profile-certificate-desired", "Wants a certificate: no.")
    own = await learn_ai_memory.learner(TOKEN, use_case="test")
    await own.put(ABOUT_KEY, "Working nurse")
    await own.put(INSTRUCTIONS_KEY, "Plain English")
    await own.put("instructions-recommendation", "Only advanced courses")
    await own.put("instructions-syllabus", "Short answers")
    return own


def scope_for(user, *, token=TOKEN):
    """Return a request scope carrying the forwarded session token."""
    headers = [(b"x-access-token", token.encode())] if token else []
    return {"user": user, "headers": headers}


async def open_notes(user, bot="recommendation", **kwargs):
    """Open the learner's notes as a consumer would."""
    return await open_learner_notes(
        scope_for(user, **kwargs), bot=bot, user_id="u1", thread_id="t1"
    )


def answers(mocker, *, durable=True, revision=None):
    """Mock the gate and extraction model calls."""

    async def ask(_self, bot_class, output_type, _instructions, prompt):
        ask.prompts.append(prompt)
        if bot_class is MemoryGateBot:
            assert output_type is GateDecision
            return GateDecision(durable=durable)
        assert bot_class is MemoryExtractionBot
        assert output_type is MemoryRevision
        return revision

    ask.prompts = []
    mocker.patch("ai_chatbots.memory.LearnerNotes._ask", ask)
    return ask


def test_session_token():
    """The forwarded token is read from the request headers."""
    assert session_token({"headers": [(b"X-Access-Token", b"abc")]}) == "abc"
    assert session_token({"headers": [(b"other", b"abc")]}) is None
    assert session_token({}) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("short", "short"),
        ("aaaa; bbbb; cccc", "aaaa; bbbb"),
        ("a" * 20, None),
    ],
)
def test_fit_length(text, expected):
    """Over the cap, whole trailing clauses are dropped, never part of one."""
    assert fit_length(text, cap=12) == expected


async def test_context_for_the_recommendation_bot(seeded, async_user):
    """The recommender sees the whole profile and its own and shared notes."""
    notes = await open_notes(async_user)
    context = notes.context
    assert context.startswith("# Learner context")
    for text in (
        "Education level: Bachelor's.",
        "Interested in: Data Science.",
        "Wants a certificate: no.",
        "Working nurse",
        "Plain English",
        "Only advanced courses",
    ):
        assert text in context
    assert "Short answers" not in context
    assert TUTOR_RULES_WIN not in context


async def test_context_for_other_bots(seeded, async_user):
    """Other bots see the education level only, and their own instructions."""
    context = (await open_notes(async_user, bot="syllabus")).context
    assert "Education level: Bachelor's." in context
    assert "Data Science" not in context
    assert "Short answers" in context
    assert "Only advanced courses" not in context


async def test_context_for_the_tutor(seeded, async_user):
    """The tutor is told its own rules win over the learner's preferences."""
    assert (await open_notes(async_user, bot="tutor")).context.endswith(TUTOR_RULES_WIN)


async def test_no_notes_no_context(learn_ai_memory, async_user):
    """A learner with no notes gets no context block."""
    assert (await open_notes(async_user)).context == ""


@pytest.mark.parametrize("case", ["no_token", "no_backend", "flag_off", "anonymous"])
async def test_memory_off(mocker, settings, learn_ai_memory, async_user, case):
    """Memory is off without a token, a backend, the flag, or a signed-in user."""
    user = async_user
    token = TOKEN
    if case == "no_token":
        token = None
    elif case == "no_backend":
        settings.OL_LLM_MEMORY_BACKEND = ""
    elif case == "flag_off":
        mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    else:
        user = None
    assert await open_notes(user, token=token) is None


async def test_memory_unavailable(mocker, learn_ai_memory, async_user, caplog):
    """A failure opening memory runs the bot without notes and logs the class."""
    mocker.patch(
        "ai_chatbots.memory.get_memory", side_effect=ValueError("secret-ish detail")
    )
    with caplog.at_level(logging.WARNING):
        assert await open_notes(async_user) is None
    assert "ValueError" in caplog.text
    assert "secret-ish detail" not in caplog.text
    assert TOKEN not in caplog.text


async def test_remember_writes_changed_notes(mocker, seeded, async_user):
    """A durable turn rewrites the notes; unchanged ones aren't written again."""
    ask = answers(
        mocker,
        revision=MemoryRevision(
            about="Working nurse; wants a data science career",
            instructions="Plain English",
            bot_instructions="",
        ),
    )
    put = mocker.spy(type(seeded), "put")
    notes = await open_notes(async_user)
    await notes.remember(
        "I want to move into data science",
        "Here are two courses <!-- {'metadata': 1} -->",
        ["earlier question"],
    )
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse; wants a data science career"
    assert stored[INSTRUCTIONS_KEY] == "Plain English"
    assert "instructions-recommendation" not in stored
    assert stored["instructions-syllabus"] == "Short answers"
    assert put.call_count == 1
    prompt = ask.prompts[0]
    assert "I want to move into data science" in prompt
    assert "earlier question" in prompt
    assert "Education level: Bachelor's." in prompt
    assert "metadata" not in prompt


async def test_remember_skips_when_the_gate_says_no(mocker, seeded, async_user):
    """Nothing durable means no rewrite and no writes."""
    ask = answers(mocker, durable=False)
    notes = await open_notes(async_user)
    await notes.remember("thanks!", "You're welcome", [])
    assert len(ask.prompts) == 1
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse"


async def test_remember_keeps_a_section_over_the_limit(
    mocker, settings, seeded, async_user
):
    """A section with no clause boundary under the cap keeps its previous text."""
    settings.AI_MEMORY_MAX_CHARS = 10
    answers(
        mocker,
        revision=MemoryRevision(
            about="x" * 50, instructions="Plain English", bot_instructions=""
        ),
    )
    await (await open_notes(async_user)).remember("m", "r", [])
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse"


async def test_remember_without_the_lock(mocker, redis_lock, seeded, async_user):
    """Another save holding the lock past the wait means no write this turn."""
    answers(
        mocker,
        revision=MemoryRevision(about="new", instructions="", bot_instructions=""),
    )
    redis_lock.acquire.return_value = False
    await (await open_notes(async_user)).remember("m", "r", [])
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse"
    redis_lock.release.assert_not_called()


async def test_remember_after_memory_is_turned_off(mocker, seeded, async_user):
    """Turning memory off during the model calls wins over the extraction."""
    answers(
        mocker,
        revision=MemoryRevision(about="new", instructions="", bot_instructions=""),
    )
    notes = await open_notes(async_user)
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    await notes.remember("m", "r", [])
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse"


async def test_remember_during_a_forget(mocker, seeded, async_user):
    """A forget in progress refuses the write, and nothing else is tried."""
    answers(
        mocker,
        revision=MemoryRevision(about="new", instructions="new", bot_instructions=""),
    )
    notes = await open_notes(async_user)
    put = mocker.patch.object(
        notes.notes, "put", side_effect=MemoryFenced("fenced", operation="put")
    )
    await notes.remember("m", "r", [])
    put.assert_called_once()


async def test_remember_never_raises(mocker, seeded, async_user, caplog):
    """A model failure after the reply is logged by class, not raised."""
    mocker.patch(
        "ai_chatbots.memory.LearnerNotes._ask", side_effect=RuntimeError("boom")
    )
    notes = await open_notes(async_user)
    with caplog.at_level(logging.WARNING):
        await notes.remember("m", "r", [])
    assert "RuntimeError" in caplog.text
    assert "boom" not in caplog.text


async def test_the_tutor_never_writes(mocker, seeded, async_user):
    """The tutor reads notes but never runs extraction."""
    ask = answers(mocker)
    await (await open_notes(async_user, bot="tutor")).remember("m", "r", [])
    assert ask.prompts == []


def test_memory_use_cases():
    """The gate and extraction calls are their own use cases on their own models."""
    assert MemoryGateBot.USE_CASE_NAME == "memory-gate"
    assert MemoryExtractionBot.USE_CASE_NAME == "memory-extraction"
    assert {"recommendation", "syllabus", "video-gpt"} == memory.WRITING_BOTS


@pytest.fixture
async def tool_call_provider(monkeypatch):
    """Serve an OpenAI-compatible stream that answers with one tool call."""
    requests = []

    async def handle(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        requests.append(body)
        tool = body["tools"][0]["function"]["name"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for delta in (
            {"tool_calls": [_tool_call(tool, '{"durable": ')]},
            {"tool_calls": [_tool_call(None, "true}")]},
        ):
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    monkeypatch.setenv("OPENAI_API_BASE", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield requests
    await runner.cleanup()


def _tool_call(name, arguments):
    call = {"index": 0, "type": "function", "function": {"arguments": arguments}}
    if name:
        call["id"] = "call_1"
        call["function"]["name"] = name
    return call


async def test_memory_bot_asks_for_structured_output(
    settings, tool_call_provider, async_user
):
    """The gate's answer comes back parsed through the policy-wrapped model."""
    settings.AI_MEMORY_GATE_MODEL = "openai/gpt-4o-mini"
    bot = await sync_to_async(MemoryGateBot)(async_user.global_id, thread_id="t1")
    answer = await bot.ask(GateDecision, "instructions", "the exchange")
    assert answer == GateDecision(durable=True)
    (request,) = tool_call_provider
    assert request["model"] == "gpt-4o-mini"
    assert request["messages"][0] == {"role": "system", "content": "instructions"}
    assert request["messages"][1]["content"] == "the exchange"
    assert request["tool_choice"]["function"]["name"] == "GateDecision"


async def test_remember_keeps_a_section_another_turn_changed(
    mocker, seeded, async_user
):
    """A section another turn rewrote since this request read it is left alone."""
    answers(
        mocker,
        revision=MemoryRevision(
            about="Working nurse; lives in Boston",
            instructions="Plain English; short answers",
            bot_instructions="Only advanced courses",
        ),
    )
    notes = await open_notes(async_user)
    await seeded.put(ABOUT_KEY, "Working nurse; wants a data science career")
    await notes.remember("m", "r", [])
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "Working nurse; wants a data science career"
    assert stored[INSTRUCTIONS_KEY] == "Plain English; short answers"


async def test_remember_when_the_lock_expired(
    mocker, redis_lock, seeded, async_user, caplog
):
    """A lock lost before release is logged; the writes that landed stand."""
    answers(
        mocker,
        revision=MemoryRevision(about="new", instructions="", bot_instructions=""),
    )
    redis_lock.release.side_effect = LockNotOwnedError("expired")
    with caplog.at_level(logging.WARNING):
        await (await open_notes(async_user)).remember("m", "r", [])
    stored = {note.key: note.text for note in await seeded.list()}
    assert stored[ABOUT_KEY] == "new"
    assert "lock expired" in caplog.text
    assert "not revised" not in caplog.text


async def test_flag_lookup_failure(mocker, learn_ai_memory, async_user):
    """A failing feature-flag lookup runs the bot without notes."""
    mocker.patch("ai_chatbots.memory.is_enabled", side_effect=ConnectionError)
    assert await open_notes(async_user) is None


async def test_open_is_bounded(mocker, settings, learn_ai_memory, async_user):
    """A memory service that hangs costs the reply at most the open timeout."""
    settings.AI_MEMORY_OPEN_TIMEOUT_SECONDS = 0.05

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    mocker.patch.object(learn_ai_memory, "learner", hang)
    assert await open_notes(async_user) is None
