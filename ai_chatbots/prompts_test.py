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


def test_support_account_tools_prompt_is_support_flavoured():
    """
    What SupportBot is actually given is the core plus its own addendum, so the
    guarantee is about the pair: it must still carry every rule for reading the
    tools, and none of the syllabus routing - it has no search tools to route
    to, and no other course's page to send anyone from.
    """
    composed = "\n\n".join(
        prompts.SYSTEM_PROMPT_MAPPING[name]
        for name in ("account_tools_core", "support_account_tools")
    )

    assert "user_course_input" in composed
    assert "couldn't check" in composed
    assert "search_content_files" not in composed
    assert "from that course's page" not in composed
    assert "resource facts" not in composed


def test_account_tools_core_holds_the_rules_both_bots_need():
    """
    The core is what any bot with these tools must obey. A rule that lives in
    only one addendum is a rule the other bot silently doesn't have.
    """
    core = " ".join(prompts.SYSTEM_PROMPT_MAPPING["account_tools_core"].split())

    assert "user_course_input" in core
    assert "exactly as they wrote it" in core
    assert "couldn't check" in core
    assert "issued" in core
    # Telling a course's terms from the learner's record is the same reading
    # job for both bots; only what each does about it differs. It also has to
    # say not to call the tools, because the answer gate is "a tool fired",
    # not "the model answered" - a lookup followed by a refusal still ships.
    assert "Will I get a certificate?" in core
    assert "don't call them for what a course offers" in core


def test_account_tools_core_names_no_bot_specific_tool():
    """
    The core is shared, so a search tool named here would be offered to a bot
    that does not have it - the model would call a tool that isn't bound.
    """
    core = prompts.SYSTEM_PROMPT_MAPPING["account_tools_core"]

    assert "search_content_files" not in core
    assert "resource facts" not in core
    assert "Contact us" not in core


def test_the_split_addenda_keep_only_what_is_theirs():
    """
    REGRESSION guard on the split itself: whatever moved to the core must not
    also be left behind, or a later edit fixes one copy and not the other.
    """
    syllabus = prompts.SYSTEM_PROMPT_MAPPING["syllabus_account_tools"]
    support = prompts.SYSTEM_PROMPT_MAPPING["support_account_tools"]

    assert "search_content_files" in syllabus
    assert "search_content_files" not in support
    # #152 made this false: the tools now take a course the learner named.
    assert "from that course's page" not in syllabus
    for addendum in (syllabus, support):
        assert "exactly as they wrote it" not in addendum
        # Both bots need this distinction, so a copy in either addendum is a
        # copy that can drift from the core's.
        assert "Will I get a certificate?" not in addendum


def test_only_the_core_says_when_to_pass_a_named_course():
    """
    REGRESSION guard on a rule that flipped: the tools used to want
    user_course_input only for a course other than the page's, and now want it
    whenever the learner names one.  An addendum is appended after the core, so
    a leftover copy of the old rule is the one the model reads last - and the
    two halves disagree with nothing failing.  The core owns this rule alone.

    Pinning the core's own sentence is deliberate: if it is reworded again,
    this test is what sends someone back to read both addenda.
    """
    core = " ".join(prompts.SYSTEM_PROMPT_MAPPING["account_tools_core"].split())

    assert "Whenever the learner names a course, pass user_course_input" in core
    for name in ("syllabus_account_tools", "support_account_tools"):
        assert "user_course_input" not in prompts.SYSTEM_PROMPT_MAPPING[name]


def test_support_is_told_what_to_do_with_a_question_about_a_courses_terms():
    """
    "Will I get a certificate?" asks what a course offers; "Did I get my
    certificate?" asks the learner's record. SyllabusBot can answer the first
    from the course's own content. SupportBot has no source for it, so its
    answer is to give none and let intake take the question - and the pair it
    is actually given has to carry both halves, the distinction and that
    resolution.
    """
    # Collapsed: these are sentences, and where the prompt happens to wrap them
    # is not the thing under test.
    composed = " ".join(
        "\n\n".join(
            prompts.SYSTEM_PROMPT_MAPPING[name]
            for name in ("account_tools_core", "support_account_tools")
        ).split()
    )

    assert "Will I get a certificate?" in composed
    assert "Did I get my certificate?" in composed
    assert "give no answer" in composed


def test_support_addendum_leaves_the_offer_to_code():
    """
    SUPPORT_OFFER_TICKET is appended to every answer by code, so a model that
    also offers makes the learner read two offers in one reply - and the
    wording of the one that matters stops being something code guarantees.
    """
    support = prompts.SYSTEM_PROMPT_MAPPING["support_account_tools"]

    assert "do not offer" in support.lower()


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
