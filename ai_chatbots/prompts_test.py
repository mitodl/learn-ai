"""Tests for open_learning_ai_tutor prompt functions w/learn_ai prompts"""

import os

import pytest
from django.core.cache import cache
from langchain_core.prompts import ChatPromptTemplate
from langsmith.utils import LangSmithNotFoundError
from open_learning_ai_tutor.prompts import get_system_prompt, langsmith_prompt_template

from ai_chatbots import chatbots, prompts
from ai_chatbots.utils import get_django_cache


def test_langsmith_prompt_create(mocker):
    """Test that the langsmith_prompt function creates a prompt if it doesn't exist."""
    os.environ["LANGSMITH_API_KEY"] = "test_key"
    os.environ["MITOL_ENVIRONMENT"] = "dev"
    mock_pull = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.pull_prompt",
        side_effect=LangSmithNotFoundError,
    )
    mock_push = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.push_prompt"
    )
    prompt = langsmith_prompt_template(
        chatbots.ResourceRecommendationBot.PROMPT_TEMPLATE,
        prompts.SYSTEM_PROMPT_MAPPING,
    )
    assert prompt.messages[0].prompt.template == prompts.PROMPT_RECOMMENDATION
    mock_pull.assert_called_once_with("recommendation_dev")
    mock_push.assert_called_once_with(
        "recommendation_dev",
        object=ChatPromptTemplate([("system", prompts.PROMPT_RECOMMENDATION)]),
    )


def test_langsmith_prompt_retrieve(mocker):
    """Test that the langsmith_prompt function retrieves a prompt."""
    os.environ["LANGSMITH_API_KEY"] = "test_key"
    os.environ["MITOL_ENVIRONMENT"] = "rc"
    mock_pull = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.pull_prompt",
        return_value=ChatPromptTemplate([("system", prompts.PROMPT_VIDEO_GPT)]),
    )
    mock_push = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.push_prompt"
    )
    prompt = langsmith_prompt_template(
        chatbots.VideoGPTBot.PROMPT_TEMPLATE, "video_gpt_rc"
    )
    mock_pull.assert_called_once_with("video_gpt_rc")
    mock_push.assert_not_called()
    assert prompt.messages[0].prompt.template == prompts.PROMPT_VIDEO_GPT


@pytest.mark.parametrize("has_cache", [True, False])
def test_get_system_prompt_no_cache(mocker, has_cache):
    """Test that the get_system_prompt function retrieves the system prompt from langsmith."""
    os.environ["LANGSMITH_API_KEY"] = "test_key"
    os.environ["MITOL_ENVIRONMENT"] = "prod"
    os.environ["CELERY_BROKER_URL"] = ""
    prompt_key = "syllabus_prod"

    def get_test_cache():
        return cache

    if has_cache:
        cache.set(prompt_key, prompts.PROMPT_SYLLABUS)
        assert cache.get(prompt_key) == prompts.PROMPT_SYLLABUS
    else:
        cache.delete(prompt_key)
        assert cache.get(prompt_key) is None
    mock_pull = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.pull_prompt",
        return_value=ChatPromptTemplate([("system", prompts.PROMPT_SYLLABUS)]),
    )
    prompt = get_system_prompt(
        chatbots.SyllabusBot.PROMPT_TEMPLATE,
        prompts.SYSTEM_PROMPT_MAPPING,
        get_test_cache,
    )
    assert mock_pull.call_count == (0 if has_cache else 1)
    assert prompt == prompts.PROMPT_SYLLABUS
    assert cache.get("syllabus_prod") == prompts.PROMPT_SYLLABUS


def test_get_system_prompt_no_langsmith(mocker):
    """get_system_prompt function should get the correct string if langsmith is not enabled"""
    os.environ["LANGSMITH_API_KEY"] = ""
    cache = get_django_cache()
    cache.clear()
    mock_pull = mocker.patch(
        "open_learning_ai_tutor.prompts.LangsmithClient.pull_prompt"
    )
    prompt = get_system_prompt(
        chatbots.SyllabusBot.PROMPT_TEMPLATE,
        prompts.SYSTEM_PROMPT_MAPPING,
        get_django_cache,
    )
    assert prompt == prompts.SYSTEM_PROMPT_MAPPING["syllabus"]
    mock_pull.assert_not_called()
