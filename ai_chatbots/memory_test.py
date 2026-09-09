"""Tests for learner memory helpers"""

import pytest
from django.contrib.auth.models import AnonymousUser

from ai_chatbots import memory
from ai_chatbots.memorystores import DjangoMemoryStore
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def flag_on(mocker):
    return mocker.patch("ai_chatbots.memory.is_enabled", return_value=True)


def test_fetch_learner_profile_returns_sample():
    """The dummy profile has the six RFC fields"""
    profile = memory.fetch_learner_profile("any-id")
    assert profile["topic_interests"][0]["name"] == "Science & Math"
    assert profile["delivery"] == ["online"]


def test_build_learner_context_renders_profile_skipping_blanks():
    """Blank fields, topic ids and preference_search_filters are left out"""
    block = memory.build_learner_context(memory.fetch_learner_profile("x"), None)
    assert "Science & Math" in block
    assert "academic-excellence" in block
    assert "online" in block
    assert "certificate" in block.lower()
    assert "731" not in block
    assert "RiTestTubeLine" not in block
    assert "preference_search_filters" not in block
    assert "time_commitment" not in block
    assert "current_education" not in block


def test_build_learner_context_includes_memory_and_truncates(settings):
    """Memory sections render under a separate heading and respect the size cap"""
    settings.AI_MEMORY_MAX_CHARS = 60
    doc = {
        "background": "Working nurse in Boston " * 20,
        "goals_and_interests": "",
        "learning_preferences": "Prefers short videos",
        "current_focus": "",
    }
    block = memory.build_learner_context({}, doc)
    assert "Working nurse" in block
    assert "Prefers short videos" not in block  # truncated away
    mem_part = block.split(memory.MEMORY_HEADING, 1)[1]
    assert len(mem_part.strip()) <= 60


def test_build_learner_context_empty():
    """Nothing to say means an empty string, not headings"""
    assert memory.build_learner_context({}, None) == ""


def test_get_learner_context_anonymous(flag_on):
    """Anonymous learners never get a context block"""
    assert memory.get_learner_context(AnonymousUser()) == ""
    assert memory.get_learner_context(None) == ""


def test_get_learner_context_flag_off(mocker):
    """Flag off means no block and no profile fetch"""
    mocker.patch("ai_chatbots.memory.is_enabled", return_value=False)
    fetch = mocker.patch("ai_chatbots.memory.fetch_learner_profile")
    assert memory.get_learner_context(UserFactory.create()) == ""
    fetch.assert_not_called()


def test_get_learner_context_reads_own_memory(flag_on):
    """The block is built from the profile and the memory of that global_id only"""
    user, other = UserFactory.create(), UserFactory.create()
    store = DjangoMemoryStore()
    store.put(
        memory.memory_namespace(user.global_id),
        "default",
        {"kind": "LearnerMemory", "content": {"current_focus": "linear algebra"}},
    )
    store.put(
        memory.memory_namespace(other.global_id),
        "default",
        {"kind": "LearnerMemory", "content": {"current_focus": "poetry"}},
    )
    block = memory.get_learner_context(user)
    assert "linear algebra" in block
    assert "poetry" not in block
    assert "Science & Math" in block
    flag_on.assert_called_once_with(
        memory.CHAT_MEMORY_FLAG, default=False, opt_unique_id=user.global_id
    )


def test_build_learner_context_starts_with_usage_instruction():
    """The block tells the bot to use it rather than ask for what's in it"""
    block = memory.build_learner_context(memory.fetch_learner_profile("x"), None)
    assert block.startswith(memory.CONTEXT_INSTRUCTION)
    assert "ask" in memory.CONTEXT_INSTRUCTION.lower()
