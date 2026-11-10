"""Tests for the ai_chatbots system checks"""

from ai_chatbots.chatbots import CanvasSyllabusBot, SyllabusBot
from ai_chatbots.checks import check_default_models

RETIRED_MODEL = "gemini/gemini-3-pro-preview"


def test_default_models_pass():
    """The configured defaults are all ones the catalog accepts."""
    assert check_default_models(None) == []


def test_retired_default_model_is_an_error(settings):
    """Every bot that falls back to a retired default is reported."""
    settings.AI_DEFAULT_SYLLABUS_MODEL = RETIRED_MODEL

    errors = check_default_models(None)

    assert {(error.id, error.obj) for error in errors} == {
        ("ai_chatbots.E001", SyllabusBot),
        ("ai_chatbots.E001", CanvasSyllabusBot),
    }
    assert all("retired" in error.msg for error in errors)


def test_retired_default_for_a_bot_that_never_calls_the_model(settings):
    """SupportBot scripts every reply, so its default is not checked."""
    settings.AI_DEFAULT_SUPPORT_MODEL = RETIRED_MODEL

    assert check_default_models(None) == []
