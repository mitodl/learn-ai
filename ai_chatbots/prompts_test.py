"""Tests for open_learning_ai_tutor prompt functions w/learn_ai prompts"""

import os

import pytest
from django.core.cache import cache
from langchain_core.prompts import ChatPromptTemplate
from langsmith.utils import LangSmithNotFoundError
from open_learning_ai_tutor.prompts import get_system_prompt, langsmith_prompt_template

from ai_chatbots import chatbots, prompts
from ai_chatbots.utils import get_django_cache


@pytest.mark.parametrize(
    "prompt", [prompts.PROMPT_SYLLABUS, prompts.PROMPT_SYLLABUS_CANVAS]
)
def test_syllabus_prompts_route_by_subject(prompt):
    """
    Both syllabus prompts should route by what a question is about: the
    resource's own facts (price included) come from its content, and the
    support center covers platform and account help.
    """
    content_paragraph, support_paragraph, tiebreak = (
        " ".join(paragraph.split())
        for paragraph in prompt.split("\n\n")
        if paragraph.startswith(
            (
                'Use "search_content_files"',
                'Use "search_support_articles"',
                "When a question involves money",
            )
        )
    )
    for topic in ("what it covers and requires", "what it costs and what it offers"):
        assert topic in content_paragraph
    for topic in (
        "something is not working",
        "already earned or paid for",
        "paying, enrolling or requesting a refund",
        "platform-wide policy",
    ):
        assert topic in support_paragraph
    # money, certificates and enrollment are the topics that were sent to the
    # support center wholesale, so the prompt splits them by subject
    assert "costs, requires or offers is in its content" in tiebreak
    # the support center has no articles about a course's captions or
    # transcripts, so those questions do not belong to it
    assert "transcript" not in support_paragraph
    assert "caption" not in support_paragraph
    # the resource facts are appended to the prompt rather than returned by a
    # tool, so the "tool output only" rule has to admit them as a source
    collapsed = " ".join(prompt.split())
    assert "from the tool output and from the resource facts below" in collapsed
    assert "OUTSIDE OF THE TOOL OUTPUT AND THE RESOURCE FACTS BELOW" in collapsed


def test_support_prompt_names_no_tools():
    """
    The base prompt is the bot's identity and its hard "never say" rules; the
    tools it happens to have are an addendum. Naming a tool here is what makes
    a second tool set a rewrite instead of one more addendum.
    """
    prompt = prompts.CHATBOT_PROMPT_MAPPING["support"]

    assert "get_enrollments" not in prompt
    assert "search_content_files" not in prompt
    assert "resource facts" not in prompt


def test_support_prompt_forbids_the_things_only_code_may_say():
    """
    Filing is deterministic and there is no human on the other end of this
    chat. A model that says a ticket was sent, or that someone will follow up,
    makes the learner stop asking - and nothing was filed.
    """
    prompt = prompts.CHATBOT_PROMPT_MAPPING["support"].lower()

    assert "never say a support ticket" in prompt
    assert "never promise" in prompt
    assert "policy" in prompt


def test_support_account_tools_addendum_is_support_flavoured():
    """
    The syllabus addendum tells the model to send the learner to another
    course's page and to fall back on search tools. Neither exists here, and
    #152 made the first one false: the tools take a course the learner named.
    """
    addendum = prompts.SYSTEM_PROMPT_MAPPING["support_account_tools"]

    assert "course_id" in addendum
    assert "couldn't check" in addendum
    assert "search_content_files" not in addendum
    assert "from that course's page" not in addendum


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
