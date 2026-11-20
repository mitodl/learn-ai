"""Tests for the code checks."""

import json

import pytest
from deepeval.test_case import LLMTestCase

from ai_chatbots.evaluation.checks import (
    bare_urls,
    ends_with_offer,
    run_checks,
    tool_use,
    unlinked_resources,
)

URL = "https://learn.mit.edu/?resource=1"


def _search(*results):
    return json.dumps({"results": list(results), "metadata": {}})


@pytest.mark.parametrize(
    ("answer", "fails"),
    [
        ("Here are two courses.\n\nIf you want, I can narrow these down.", True),
        # Curly apostrophe
        ("Here they are.\n\nIf you\u2019d like, I can compare them.", True),
        ("Here they are.\n\nWould you like more options?", True),
        # A trailing rule or comment is not the last paragraph
        ("Here they are.\n\nLet me know if that helps.\n\n---", True),
        # gpt-4o-mini's no-results ending, with no "if" after "let me know"
        (
            "Here they are.\n\nIf you're looking for something more specific or "
            "different, please let me know!",
            True,
        ),
        ("Here they are.\n\nBoth are free.", False),
        # An offer earlier in the answer doesn't count
        ("If you want a certificate, pick the first.\n\nBoth start in May.", False),
        # "If you want" describing a fit, not offering more
        ("1. [A](u)\n\n2. [B](v)\n   Best fit if you want modern history.", False),
        ("It's free; only pay if you want the certificate track.", False),
        ("Here they are.\n\nIf you'd like more options, just ask.", True),
        ("Here they are.\n\nI can also format it as an exam if you want.", True),
    ],
)
def test_ends_with_offer(answer, fails):
    """Only an offer in the last paragraph fails."""
    assert bool(ends_with_offer(answer)) is fails


@pytest.mark.parametrize(
    ("answer", "unlinked"),
    [
        (f"Try [Intro to Data Science]({URL}).", []),
        # Scheme/host case and a trailing slash don't matter
        ("Try [Intro to Data Science](HTTPS://Learn.mit.edu/?resource=1).", []),
        ("Try Intro to Data Science.", ["Intro to Data Science"]),
        # Linked, but to the wrong place
        (
            "Try [Intro to Data Science](https://learn.mit.edu/?resource=2).",
            ["Intro to Data Science"],
        ),
        ("Try [Intro to Data Science](#).", ["Intro to Data Science"]),
        ("Try [Intro to Data Science](https://[broken).", ["Intro to Data Science"]),
        # Not mentioned at all
        ("Try something else.", []),
    ],
)
def test_unlinked_resources(answer, unlinked):
    """A returned title must be the text of a link to its own url."""
    context = [_search({"title": "Intro to Data Science", "url": URL})]
    assert unlinked_resources(answer, context) == unlinked


def test_unlinked_resources_shared_titles():
    """Any result's url links a shared title; a longer title doesn't mention it."""
    context = [
        _search(
            {"title": "Calculus", "url": URL},
            {"title": "Calculus", "url": f"{URL}0"},
            {"title": "Calculus for Poets", "url": f"{URL}1"},
        )
    ]
    assert unlinked_resources(f"Try [Calculus]({URL}0).", context) == []
    assert unlinked_resources(f"Try [Calculus for Poets]({URL}1).", context) == []
    assert unlinked_resources("Try Calculus.", context) == ["Calculus"]


@pytest.mark.parametrize(
    ("answer", "found"),
    [
        (f"Try [Intro]({URL}).", []),
        (f"Try {URL} for this.", [URL]),
        (f"Try [{URL}]({URL}).", [URL]),
        ("Search MIT Learn for more.", []),
    ],
)
def test_bare_urls(answer, found):
    """A url outside a link, or as a link's text, fails."""
    assert bare_urls(answer) == found


@pytest.mark.parametrize(
    ("bot", "checks"),
    [
        (
            "recommendation",
            {"follow_up_offer", "unlinked_resource", "bare_url", "tool_use"},
        ),
        ("syllabus", {"follow_up_offer"}),
        ("syllabus_canvas", {"follow_up_offer"}),
        ("video_gpt", {"follow_up_offer"}),
        ("tutor", set()),
    ],
)
def test_run_checks_by_bot(bot, checks):
    """Each bot gets only the checks that apply to it."""
    case = LLMTestCase(
        input="q",
        actual_output="Answer.\n\nWould you like more?",
        retrieval_context=[],
        metadata={},
    )
    found = run_checks(bot, case)
    assert set(found) == checks
    if checks:
        assert found["follow_up_offer"]


@pytest.mark.parametrize(
    ("expected", "called", "found"),
    [
        # A case that doesn't say which tools it expects isn't checked
        (None, [], []),
        (None, ["search_courses"], []),
        ([], [], []),
        ([], ["search_courses", "search_courses"], ["called search_courses"]),
        (["search_courses"], ["search_courses"], []),
        # Other tools alongside the expected one are fine
        (["search_courses"], ["search_content_files", "search_courses"], []),
        (["search_courses"], [], ["didn't call search_courses"]),
    ],
)
def test_tool_use(expected, called, found):
    """Flag a missing expected tool, or any tool call when none was expected."""
    assert tool_use({"expected_tools": expected, "tool_names": called}) == found
