"""Tests for learner memory helpers"""

import pytest
from django.contrib.auth.models import AnonymousUser

from ai_chatbots import memory, prompts
from ai_chatbots.memorystores import DjangoMemoryStore
from main.factories import UserFactory

pytestmark = pytest.mark.django_db

REC = "ResourceRecommendationBot"


@pytest.fixture
def flag_on(mocker):
    return mocker.patch("ai_chatbots.memory.is_enabled", return_value=True)


def test_fetch_learner_profile_returns_sample():
    """The dummy profile has the six RFC fields"""
    profile = memory.fetch_learner_profile("any-id")
    assert profile["topic_interests"][0]["name"] == "Science & Math"
    assert profile["delivery"] == ["online"]


def test_recommendation_block_renders_full_profile_skipping_blanks():
    """The recommendation bot sees every usable profile field, not ids or filters"""
    block = memory.build_learner_context(
        REC, memory.fetch_learner_profile("x"), memory.LearnerMemory()
    )
    assert block.startswith(memory.CONTEXT_INSTRUCTION)
    assert "Science & Math" in block
    assert "academic-excellence" in block
    assert "online" in block
    assert "free, non-certificate" in block
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
    [("no", "free, non-certificate"), ("yes", "paid certificate"), ("", None)],
)
def test_certificate_preference_implies_price(certificate_desired, expected):
    """Certificate tracks are the paid ones, so the block spells out the price implication"""
    block = memory.build_learner_context(
        REC, {"certificate_desired": certificate_desired}, memory.LearnerMemory()
    )
    if expected is None:
        assert block == ""
    else:
        assert expected in block


def test_block_renders_about_and_instructions_sections(settings):
    """Shared facts, shared instructions and this bot's instructions; capped per section"""
    settings.AI_MEMORY_MAX_CHARS = 40
    mem = memory.LearnerMemory(
        about="Working nurse in Boston. " * 5,
        instructions="Keep answers short.",
        bot_instructions="Never show social science courses.",
    )
    block = memory.build_learner_context(REC, {}, mem)
    assert memory.ABOUT_HEADING in block
    assert memory.INSTRUCTIONS_HEADING in block
    assert "Keep answers short." in block
    assert "Never show social science courses." in block
    about = block.split(memory.ABOUT_HEADING, 1)[1].split(memory.INSTRUCTIONS_HEADING)[
        0
    ]
    assert len(about.strip()) <= 40


def test_block_empty_when_nothing_known():
    """Nothing to say means an empty string, not headings"""
    assert memory.build_learner_context(REC, {}, memory.LearnerMemory()) == ""


def test_get_learner_context_anonymous(flag_on):
    """Anonymous learners never get a context block"""
    assert memory.get_learner_context(AnonymousUser(), REC) == ""
    assert memory.get_learner_context(None, REC) == ""


def test_get_learner_context_flag_off(mocker):
    """Flag off means no block and no profile fetch"""
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    fetch = mocker.patch("ai_chatbots.memory.fetch_learner_profile")
    assert memory.get_learner_context(UserFactory.create(), REC) == ""
    fetch.assert_not_called()


def test_get_learner_context_reads_own_keys_only(flag_on):
    """Shared about + shared instructions + this bot's instructions, this learner only"""
    user, other = UserFactory.create(), UserFactory.create()
    memory.save_learner_memory(
        user.global_id,
        REC,
        memory.LearnerMemory(
            about="nurse",
            instructions="short answers",
            bot_instructions="no social sci",
        ),
    )
    memory.save_learner_memory(
        user.global_id, "TutorBot", memory.LearnerMemory(bot_instructions="hints only")
    )
    memory.save_learner_memory(other.global_id, REC, memory.LearnerMemory(about="poet"))
    block = memory.get_learner_context(user, REC)
    assert "nurse" in block
    assert "short answers" in block
    assert "no social sci" in block
    assert "hints only" not in block
    assert "poet" not in block
    tutor_block = memory.get_learner_context(user, "TutorBot")
    assert "hints only" in tutor_block
    assert "no social sci" not in tutor_block
    flag_on.assert_called_with(
        memory.CHAT_MEMORY_FLAG, default=False, opt_unique_id=user.global_id
    )


def test_save_learner_memory_writes_only_changed_sections():
    """Unchanged or empty sections don't touch the store"""
    store = DjangoMemoryStore()
    memory.save_learner_memory(
        "g1", REC, memory.LearnerMemory(about="nurse", bot_instructions="no social sci")
    )
    ns = memory.memory_namespace("g1")
    assert sorted(i.key for i in store.search(ns)) == [
        memory.ABOUT_KEY,
        memory.bot_instructions_key(REC),
    ]
    first = store.get(ns, memory.ABOUT_KEY).updated_at
    memory.save_learner_memory(
        "g1", REC, memory.LearnerMemory(about="nurse", bot_instructions="hints")
    )
    assert store.get(ns, memory.ABOUT_KEY).updated_at == first
    assert store.get(ns, memory.bot_instructions_key(REC)).value["text"] == "hints"


def test_load_learner_memory_roundtrip():
    """Loading returns what was saved, missing keys as empty strings"""
    memory.save_learner_memory("g2", REC, memory.LearnerMemory(about="nurse"))
    mem = memory.load_learner_memory("g2", REC)
    assert mem == memory.LearnerMemory(
        about="nurse", instructions="", bot_instructions=""
    )


def test_extract_learner_memory_passes_context_and_saves(mocker, settings):
    """One structured call gets bot name, current memory and the exchange; result is saved"""
    settings.AI_MEMORY_MAX_CHARS = 1500
    memory.save_learner_memory("g3", REC, memory.LearnerMemory(about="nurse"))
    llm = mocker.patch("ai_chatbots.memory.init_chat_model").return_value
    structured = llm.with_structured_output.return_value
    structured.invoke.return_value = memory.LearnerMemory(
        about="nurse in Boston",
        instructions="",
        bot_instructions="avoid social science",
    )
    memory.extract_learner_memory(
        "g3", REC, "I'm in Boston and not a social scientist", "Noted."
    )
    llm.with_structured_output.assert_called_once_with(memory.LearnerMemory)
    prompt_text = "".join(m.content for m in structured.invoke.call_args.args[0])
    assert REC in prompt_text
    assert memory.BOT_PURPOSE[REC] in prompt_text
    assert "nurse" in prompt_text  # current memory shown
    assert "I'm in Boston and not a social scientist" in prompt_text
    assert "Noted." in prompt_text
    mem = memory.load_learner_memory("g3", REC)
    assert mem.about == "nurse in Boston"
    assert mem.bot_instructions == "avoid social science"


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
    ):
        assert phrase in text


def test_recommendation_prompt_defers_to_learner_context():
    """The static prompt must not tell the bot to re-ask what the learner context answers"""
    assert "Learner context" in prompts.PROMPT_RECOMMENDATION
    assert "If the user's intent is unclear, ask clarifying" not in (
        prompts.PROMPT_RECOMMENDATION
    )
    assert (
        "search" in prompts.PROMPT_RECOMMENDATION.split("Learner context", 1)[1][:400]
    )
