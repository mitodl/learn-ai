"""Tests for learner memory helpers"""

from datetime import timedelta

import pytest
from django.contrib.auth.models import AnonymousUser
from django.utils import timezone

from ai_chatbots import memory, prompts
from ai_chatbots.factories import CheckpointFactory, UserChatSessionFactory
from ai_chatbots.models import LearnerMemoryNote, LearnerMemoryState, PendingMemoryTurn
from main.factories import UserFactory

pytestmark = pytest.mark.django_db

REC = "ResourceRecommendationBot"
TUTOR = "TutorBot"


@pytest.fixture
def flag_on(mocker):
    return mocker.patch("ai_chatbots.memory.is_enabled", return_value=True)


@pytest.fixture
def redis_lock(mocker):
    """Fake per-user lock; set .acquired to False to simulate contention."""
    lock = mocker.Mock()
    lock.acquire.side_effect = lambda **_: lock.acquired
    lock.acquired = True
    mocker.patch(
        "ai_chatbots.memory.get_redis_connection"
    ).return_value.lock.return_value = lock
    return lock


@pytest.fixture
def user():
    return UserFactory.create()


def _mock_models(mocker, *, gate=True, revision=None):
    """Patch init_chat_model so the gate and extractor return canned results."""
    init = mocker.patch("ai_chatbots.memory.init_chat_model")
    structured = init.return_value.with_structured_output.return_value

    def invoke(messages):
        schema = init.return_value.with_structured_output.call_args.args[0]
        if schema is memory.GateDecision:
            return memory.GateDecision(durable=gate)
        return revision or memory.MemoryRevision()

    structured.invoke.side_effect = invoke
    return init, structured


# --- rendering -----------------------------------------------------------------


def test_recommendation_block_renders_full_profile_skipping_blanks():
    """The recommendation bot sees every usable profile field, not ids or filters"""
    block = memory.build_learner_context(
        REC, memory.fetch_learner_profile("x"), memory.LearnerMemory()
    )
    assert block.startswith(memory.CONTEXT_INSTRUCTION)
    assert memory.PROFILE_HEADING in block
    assert "Science & Math" in block
    assert "academic-excellence" in block
    assert "online" in block
    assert "do not ask about price" in block
    assert "731" not in block
    assert "preference_search_filters" not in block
    assert "time_commitment" not in block


def test_content_bots_only_see_background_from_profile():
    """Inside a course, topic and delivery preferences are noise"""
    profile = {**memory.fetch_learner_profile("x"), "current_education": "Bachelor's"}
    block = memory.build_learner_context("SyllabusBot", profile, memory.LearnerMemory())
    assert "Bachelor's" in block
    assert "Science & Math" not in block
    assert "online" not in block
    assert "certificate" not in block.lower()


@pytest.mark.parametrize(
    ("certificate_desired", "expected"),
    [("no", "do not ask about price"), ("yes", "wants one"), ("", None)],
)
def test_certificate_rendering_is_policy_not_budget(certificate_desired, expected):
    """Declining a certificate suppresses the price question; it never claims 'free'"""
    block = memory.build_learner_context(
        REC, {"certificate_desired": certificate_desired}, memory.LearnerMemory()
    )
    if expected is None:
        assert block == ""
    else:
        assert expected in block
        assert "free" not in block


def test_block_labels_profile_and_learned_notes_separately(settings):
    """Stated profile and learned notes are distinguishable; sections are capped"""
    settings.AI_MEMORY_MAX_CHARS = 40
    mem = memory.LearnerMemory(
        about="Working nurse in Boston. " * 5,
        instructions="Keep answers short.",
        bot_instructions="Never show social science courses.",
    )
    block = memory.build_learner_context(REC, {"current_education": "BSc"}, mem)
    assert block.index(memory.PROFILE_HEADING) < block.index(memory.ABOUT_HEADING)
    assert memory.INSTRUCTIONS_HEADING in block
    assert "Keep answers short." in block
    assert "Never show social science courses." in block
    about = block.split(memory.ABOUT_HEADING, 1)[1].split(memory.INSTRUCTIONS_HEADING)[
        0
    ]
    assert len(about.strip()) <= 40
    assert about.strip().endswith("Boston")


def test_block_empty_when_nothing_known():
    """Nothing to say means an empty string, not headings"""
    assert memory.build_learner_context(REC, {}, memory.LearnerMemory()) == ""


def test_context_instruction_marks_notes_as_learner_data():
    """Memory is lower-trust data: app rules win, then the live conversation"""
    text = memory.CONTEXT_INSTRUCTION
    assert "learner-supplied data" in text
    assert "take precedence" in text
    assert "this conversation" in text
    assert "override" in text


def test_fit_length_cuts_at_a_clause_boundary():
    assert memory.fit_length("short", 40) == "short"
    assert memory.fit_length("Nurse in Boston; wants data science; likes cats", 38) == (
        "Nurse in Boston; wants data science"
    )
    assert memory.fit_length("x" * 50, 10) == "x" * 10


def test_get_learner_context_anonymous(flag_on):
    assert memory.get_learner_context(AnonymousUser(), REC) == ""
    assert memory.get_learner_context(None, REC) == ""


def test_get_learner_context_flag_off(mocker, user):
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    fetch = mocker.patch("ai_chatbots.memory.fetch_learner_profile")
    assert memory.get_learner_context(user, REC) == ""
    fetch.assert_not_called()


def test_get_learner_context_profile_failure_keeps_notes(flag_on, mocker, user):
    """A profile fetch error must not hide the learned notes"""
    mocker.patch("ai_chatbots.memory.fetch_learner_profile", side_effect=OSError)
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse"})
    block = memory.get_learner_context(user, REC)
    assert "nurse" in block
    assert memory.PROFILE_HEADING not in block


def test_get_learner_context_reads_own_keys_only(flag_on, user):
    """Shared about + shared instructions + this bot's instructions, this learner only"""
    other = UserFactory.create()
    memory.save_notes(
        user,
        {
            memory.ABOUT_KEY: "nurse",
            memory.INSTRUCTIONS_KEY: "short answers",
            memory.bot_instructions_key(REC): "no social sci",
            memory.bot_instructions_key(TUTOR): "hints only",
        },
    )
    memory.save_notes(other, {memory.ABOUT_KEY: "poet"})
    block = memory.get_learner_context(user, REC)
    assert "nurse" in block
    assert "short answers" in block
    assert "no social sci" in block
    assert "hints only" not in block
    assert "poet" not in block
    tutor_block = memory.get_learner_context(user, TUTOR)
    assert "hints only" in tutor_block
    assert "no social sci" not in tutor_block
    flag_on.assert_called_with(
        memory.CHAT_MEMORY_FLAG, default=False, opt_unique_id=user.global_id
    )


# --- notes -----------------------------------------------------------------------


def test_save_notes_writes_changed_and_deletes_emptied_sections(user):
    """Unchanged sections are untouched; an empty revision removes the note"""
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse", "instructions:X": "no social"})
    first = LearnerMemoryNote.objects.get(user=user, key=memory.ABOUT_KEY).updated_on
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse", "instructions:X": ""})
    assert (
        LearnerMemoryNote.objects.get(user=user, key=memory.ABOUT_KEY).updated_on
        == first
    )
    assert not LearnerMemoryNote.objects.filter(
        user=user, key="instructions:X"
    ).exists()


def test_notes_cascade_with_user(user):
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse"})
    memory.record_memory_turn(user, REC, "t", "hi", "hello", 0)
    user.delete()
    assert not LearnerMemoryNote.objects.exists()
    assert not PendingMemoryTurn.objects.exists()
    assert not LearnerMemoryState.objects.exists()


# --- pending turns and clearing --------------------------------------------------


def test_record_memory_turn_first_row_schedules_later_rows_join(user, settings):
    settings.AI_MEMORY_REPLY_CHARS = 5
    assert memory.record_memory_turn(user, REC, "t-1", "hi", "hello world", 0) is True
    assert memory.record_memory_turn(user, TUTOR, "t-2", "help", "sure", 0) is False
    rows = list(PendingMemoryTurn.objects.order_by("id"))
    assert [(r.bot, r.response) for r in rows] == [(REC, "hello"), (TUTOR, "sure")]
    assert LearnerMemoryState.objects.get(user=user).generation == 0


def test_record_memory_turn_discards_pre_clear_exchange(user, redis_lock):
    """A response that started before a clear must not become memory"""
    memory.clear_learner_memory(user)
    assert memory.record_memory_turn(user, REC, "t-1", "hi", "hello", 0) is False
    assert not PendingMemoryTurn.objects.exists()
    assert memory.record_memory_turn(user, REC, "t-1", "hi", "hello", 1) is True


def test_clear_learner_memory_wipes_and_bumps_generation(user, redis_lock):
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse"})
    memory.record_memory_turn(user, REC, "t", "hi", "hello", 0)
    memory.clear_learner_memory(user)
    state = LearnerMemoryState.objects.get(user=user)
    assert state.generation == 1
    assert state.cleared_at is not None
    assert not LearnerMemoryNote.objects.filter(user=user).exists()
    assert not PendingMemoryTurn.objects.filter(user=user).exists()
    redis_lock.acquire.assert_called_once_with(blocking=True)


def test_clear_learner_memory_busy_when_lock_held(user, redis_lock):
    redis_lock.acquired = False
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse"})
    with pytest.raises(memory.MemoryBusy):
        memory.clear_learner_memory(user)
    assert LearnerMemoryNote.objects.filter(user=user).exists()


def test_stale_users_and_stuck_counts(user, settings):
    settings.AI_MEMORY_MAX_ATTEMPTS = 2
    old = timezone.now() - timedelta(seconds=settings.AI_MEMORY_DELAY_SECONDS + 60)
    memory.record_memory_turn(user, REC, "t", "old", "r", 0)
    PendingMemoryTurn.objects.update(created_on=old)
    memory.record_memory_turn(user, REC, "t", "new", "r", 0)
    other = UserFactory.create()
    memory.record_memory_turn(other, REC, "t", "stuck", "r", 0)
    PendingMemoryTurn.objects.filter(user=other).update(created_on=old, attempts=2)
    assert memory.stale_users() == [user.id]
    assert memory.stuck_turn_count() == 1


# --- history ---------------------------------------------------------------------


def _lc(kind, content):
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", kind],
        "kwargs": {"type": kind.replace("Message", "").lower(), "content": content},
    }


def _thread_with_history(user, thread_id, *contents):
    session = UserChatSessionFactory.create(user=user, thread_id=thread_id)
    cp = CheckpointFactory.create(thread_id=thread_id, session=session)
    cp.checkpoint = {
        "channel_values": {
            "messages": [
                _lc("HumanMessage" if i % 2 == 0 else "AIMessage", c)
                for i, c in enumerate(contents)
            ]
        }
    }
    cp.save()
    return session


def test_thread_history_returns_earlier_learner_messages_from_own_thread(user):
    """Learner turns before the one being processed, from the newest checkpoint"""
    _thread_with_history(
        user,
        "t-1",
        "find ecology courses",
        "here",
        "I'm a beginner on this topic",
        "ok",
    )
    assert memory.thread_history(user, "t-1", "I'm a beginner on this topic", None) == [
        "find ecology courses"
    ]
    assert memory.thread_history(user, "t-1", "unrelated", None) == [
        "find ecology courses",
        "I'm a beginner on this topic",
    ]
    assert memory.thread_history(UserFactory.create(), "t-1", "x", None) == []
    assert memory.thread_history(user, "no-such-thread", "x", None) == []


def test_thread_history_excluded_for_threads_that_predate_a_clear(user):
    """Old threads could reintroduce forgotten facts, so they contribute no context"""
    session = _thread_with_history(user, "t-1", "I'm a nurse", "ok", "beginner here")
    after = session.created_on + timedelta(minutes=1)
    before = session.created_on - timedelta(minutes=1)
    assert memory.thread_history(user, "t-1", "beginner here", after) == []
    assert memory.thread_history(user, "t-1", "beginner here", before) == [
        "I'm a nurse"
    ]


def test_render_batch_labels_bots_and_threads(user):
    _thread_with_history(user, "t-1", "find ecology courses", "here", "beginner", "ok")
    memory.record_memory_turn(user, REC, "t-1", "beginner", "Noted.", 0)
    memory.record_memory_turn(user, TUTOR, "t-2", "hints only please", "Sure.", 0)
    batch = list(PendingMemoryTurn.objects.order_by("id"))
    text = memory.render_batch(user, batch, None)
    assert "context only, already reflected" in text
    assert "- find ecology courses" in text
    assert f"[{REC}, thread 1] Learner: beginner" in text
    assert f"[{TUTOR}, thread 2] Learner: hints only please" in text
    assert "Chatbot reply (context only): Sure." in text


# --- extraction ------------------------------------------------------------------


def test_worth_extracting_asks_the_gate_model_with_notes(mocker, settings):
    settings.AI_MEMORY_GATE_MODEL = "openai:cheap"
    settings.AI_MEMORY_LLM_TIMEOUT = 7
    init, structured = _mock_models(mocker, gate=False)
    notes = {"instructions:X": "only introductory ecology"}
    assert memory.worth_extracting("[X] Learner: find ecology courses", notes) is False
    init.assert_called_once_with(
        "openai:cheap", temperature=0, timeout=7, max_retries=1
    )
    prompt_text = "".join(m.content for m in structured.invoke.call_args.args[0])
    assert "only introductory ecology" in prompt_text
    assert "find ecology courses" in prompt_text


def test_process_learner_memory_saves_batch_and_deletes_rows(
    mocker, flag_on, user, redis_lock
):
    """Gate yes: one rewrite over the batch, notes saved, only those rows removed"""
    _thread_with_history(user, "t-1", "find ecology courses", "here", "beginner", "ok")
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse", "instructions:Other": "keep"})
    memory.record_memory_turn(user, REC, "t-1", "beginner", "Noted.", 0)
    memory.record_memory_turn(user, TUTOR, "t-2", "hints only", "Sure.", 0)
    revision = memory.MemoryRevision(
        about="nurse in Boston",
        instructions="",
        bot_instructions=[
            memory.BotInstructions(bot=REC, instructions="beginner in ecology"),
            memory.BotInstructions(bot=TUTOR, instructions="hints only"),
            memory.BotInstructions(bot="Other", instructions="ignored: not in batch"),
        ],
    )
    _, structured = _mock_models(mocker, gate=True, revision=revision)

    assert memory.process_learner_memory(user.id) == "saved"

    prompt_text = "".join(m.content for m in structured.invoke.call_args.args[0])
    assert memory.BOT_PURPOSE[REC] in prompt_text
    assert memory.BOT_PURPOSE[TUTOR] in prompt_text
    assert '"about": "nurse"' in prompt_text
    assert "- find ecology courses" in prompt_text
    assert "Learner: hints only" in prompt_text
    notes = memory.load_notes(user)
    assert notes == {
        memory.ABOUT_KEY: "nurse in Boston",
        f"instructions:{REC}": "beginner in ecology",
        f"instructions:{TUTOR}": "hints only",
        "instructions:Other": "keep",
    }
    assert not PendingMemoryTurn.objects.exists()


def test_process_learner_memory_gate_no_drops_rows_keeps_notes(
    mocker, flag_on, user, redis_lock
):
    memory.save_notes(user, {memory.ABOUT_KEY: "nurse"})
    memory.record_memory_turn(user, REC, "t-1", "find ecology courses", "Here.", 0)
    init, _ = _mock_models(mocker, gate=False)
    assert memory.process_learner_memory(user.id) == "skipped"
    assert init.call_count == 1
    assert memory.load_notes(user) == {memory.ABOUT_KEY: "nurse"}
    assert not PendingMemoryTurn.objects.exists()


def test_process_learner_memory_lock_held_leaves_rows(
    mocker, flag_on, user, redis_lock
):
    redis_lock.acquired = False
    memory.record_memory_turn(user, REC, "t-1", "I'm a nurse", "ok", 0)
    init = mocker.patch("ai_chatbots.memory.init_chat_model")
    assert memory.process_learner_memory(user.id) == "locked"
    init.assert_not_called()
    assert PendingMemoryTurn.objects.filter(user=user, attempts=0).count() == 1


def test_process_learner_memory_clear_during_extraction_discards(
    mocker, flag_on, user, redis_lock
):
    """A clear between the model calls and the commit wins"""
    memory.record_memory_turn(user, REC, "t-1", "I'm a nurse", "ok", 0)
    revision = memory.MemoryRevision(about="nurse")
    _mock_models(mocker, gate=True, revision=revision)

    def clear_then_gate(*_a, **_k):
        LearnerMemoryState.objects.filter(user=user).update(generation=1)
        PendingMemoryTurn.objects.filter(user=user).delete()
        return True

    mocker.patch("ai_chatbots.memory.worth_extracting", side_effect=clear_then_gate)
    assert memory.process_learner_memory(user.id) == "cleared"
    assert not LearnerMemoryNote.objects.filter(user=user).exists()


def test_process_learner_memory_respects_batch_limits(
    mocker, flag_on, user, redis_lock, settings
):
    """Oldest rows first, up to the count and size limits; the rest stay pending"""
    settings.AI_MEMORY_BATCH_SIZE = 2
    for i in range(3):
        memory.record_memory_turn(user, REC, "t", f"m{i}", "r", 0)
    _mock_models(mocker, gate=False)
    assert memory.process_learner_memory(user.id) == "skipped"
    assert list(PendingMemoryTurn.objects.values_list("message", flat=True)) == ["m2"]
    settings.AI_MEMORY_BATCH_SIZE = 10
    settings.AI_MEMORY_BATCH_CHARS = 3
    memory.record_memory_turn(user, REC, "t", "m3", "r", 0)
    assert memory.process_learner_memory(user.id) == "skipped"
    assert list(PendingMemoryTurn.objects.values_list("message", flat=True)) == ["m3"]


def test_process_learner_memory_increments_attempts_and_skips_stuck(
    mocker, flag_on, user, redis_lock, settings
):
    settings.AI_MEMORY_MAX_ATTEMPTS = 1
    memory.record_memory_turn(user, REC, "t", "m", "r", 0)
    mocker.patch("ai_chatbots.memory.worth_extracting", side_effect=RuntimeError)
    with pytest.raises(RuntimeError):
        memory.process_learner_memory(user.id)
    assert PendingMemoryTurn.objects.get().attempts == 1
    assert memory.process_learner_memory(user.id) == "empty"


def test_process_learner_memory_flag_off_drops_backlog(mocker, user, redis_lock):
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    memory.record_memory_turn(user, REC, "t", "m", "r", 0)
    assert memory.process_learner_memory(user.id) == "disabled"
    assert not PendingMemoryTurn.objects.exists()
    redis_lock.acquire.assert_not_called()


def test_process_learner_memory_unknown_user():
    assert memory.process_learner_memory(0) == "no-user"


def test_extraction_prompt_has_the_guard_rails():
    """Rules the extraction prompt must state, given the assessment-content history"""
    text = memory.EXTRACTION_INSTRUCTIONS.lower()
    for phrase in (
        "answer",
        "grade",
        "problem",
        "assistant",
        "learner's own",
        "a question is never a fact",
        "different chatbot",
        "do not append",
        "keep the scope",
        "current preference only",
        "complaint",
        "this time",
        "remove that restriction",
        "rules, tools, permissions",
    ):
        assert phrase in text


def test_gate_prompt_covers_remember_that_and_temporary_requests():
    text = memory.GATE_INSTRUCTIONS.lower()
    assert "remember something" in text
    assert "only to this one" in text


def test_recommendation_prompt_defers_to_learner_context():
    """The static prompt must not tell the bot to re-ask what the learner context answers"""
    assert "Learner context" in prompts.PROMPT_RECOMMENDATION
    assert "If the user's intent is unclear, ask clarifying" not in (
        prompts.PROMPT_RECOMMENDATION
    )
    assert (
        "search" in prompts.PROMPT_RECOMMENDATION.split("Learner context", 1)[1][:400]
    )
    assert "before saying" in prompts.PROMPT_RECOMMENDATION
