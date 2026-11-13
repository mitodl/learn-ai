"""Tests for the ai_chatbots system checks"""

from ai_chatbots.chatbots import CanvasSyllabusBot, SupportBot, SyllabusBot
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


def test_retired_support_default_is_an_error(settings):
    """
    SupportBot scripts most of its replies, so SETS_UP_THE_MODEL_EAGERLY is
    False and a retired default costs a learner the answer and drops them into
    intake rather than failing the request. That silence is why the deploy check
    has to cover this bot: with intake still working, nothing else reports it.
    The check reads CALLS_THE_MODEL off the class, where the per-request value
    is not visible, so this bot has to opt in unconditionally.
    """
    settings.AI_DEFAULT_SUPPORT_MODEL = RETIRED_MODEL

    errors = check_default_models(None)

    # Membership, not the whole set: any other bot whose default is blank or
    # retired in this environment is reported too, and that is not this test's
    # business.
    assert ("ai_chatbots.E001", SupportBot) in {
        (error.id, error.obj) for error in errors
    }
