"""Tests for comparing a candidate with the baseline."""

import pytest

from ai_chatbots.evaluation.compare import (
    measure_outcomes,
    print_comparison,
    sign_test_p,
)

BASE = "openai/base-model"
CAND = "openai/candidate-model"


def _records(n, candidate_offers=()):
    """Answer records for n cases; the candidate ends some with an offer."""
    return [
        {
            "case_id": f"f#{i}",
            "model": model,
            "question": f"Question {i}",
            "checks": {
                "follow_up_offer": ["offer"]
                if model == CAND and i in candidate_offers
                else []
            },
        }
        for i in range(n)
        for model in (BASE, CAND)
    ]


def _output(write):
    return "\n".join(str(c.args[0]) for c in write.call_args_list)


def test_sign_test_p():
    """Exact two-sided binomial p, ignoring ties."""
    assert sign_test_p(6, 0) == pytest.approx(2 / 64)
    assert sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sign_test_p(0, 0) == 1.0


def test_measure_outcomes():
    """
    Each measure gives better, worse or same per case.  Unjudged cases, skipped
    ones and one only the baseline answered are left out.
    """
    scores = {
        ("f#0", BASE): 0.9,
        ("f#0", CAND): 0.5,  # worse
        ("f#1", BASE): 0.4,
        ("f#1", CAND): 0.8,  # better
        ("f#2", BASE): 0.9,
        ("f#2", CAND): 0.9,  # same
        ("f#3", BASE): 0.9,  # the candidate's score is a judge error
    }
    winners = {"f#0": BASE, "f#1": CAND, "f#2": None}
    records = _records(6, candidate_offers={0})[:-1]  # f#5: baseline only
    outcomes, questions = measure_outcomes(
        records, scores, winners, BASE, CAND, skip={"f#4"}
    )
    assert outcomes == {
        "pairwise": {"f#0": "worse", "f#1": "better"},
        "Correctness": {"f#0": "worse", "f#1": "better", "f#2": "same"},
        "follow_up_offer": {
            "f#0": "worse",
            "f#1": "same",
            "f#2": "same",
            "f#3": "same",
        },
    }
    assert list(questions) == ["f#0", "f#1", "f#2", "f#3"]


@pytest.mark.parametrize(
    ("worse", "better", "expected"),
    [
        (6, 0, "CLEARLY WORSE"),
        (10, 2, "CLEARLY WORSE"),
        (5, 0, "too few cases to flag"),
        (7, 1, "not flagged"),  # p = 0.07
        (0, 6, "not flagged"),  # the candidate is better
    ],
)
def test_print_comparison_flags(mocker, worse, better, expected):
    """Flagged when worse outnumbers better and the sign test gives p < 0.05."""
    write = mocker.Mock()
    by_case = {f"c{i}": "worse" for i in range(worse)}
    by_case |= {f"d{i}": "better" for i in range(better)}
    questions = dict.fromkeys(by_case, "Question")

    flagged = print_comparison(
        write, {"follow_up_offer": by_case}, questions, failed=False
    )

    output = _output(write)
    assert expected in output
    is_flagged = expected == "CLEARLY WORSE"
    assert flagged == (["follow_up_offer"] if is_flagged else [])
    assert ("Regression found" in output) is is_flagged
    assert ("No regression found" in output) is not is_flagged
    # The cases behind a flagged code check are listed
    assert ("↳ worse: c0: Question" in output) is is_flagged


@pytest.mark.parametrize(
    ("questions", "failed", "why"),
    [({"c0": "Q"}, True, "see FAILURES"), ({}, False, "no cases compared")],
)
def test_print_comparison_incomplete(mocker, questions, failed, why):
    """A failure, or nothing compared, is Incomplete whatever the counts say."""
    write = mocker.Mock()
    worse = {f"c{i}": "worse" for i in range(8)}
    print_comparison(write, {"pairwise": worse}, questions, failed=failed)
    output = _output(write)
    assert f"Incomplete ({why})" in output
    assert "Regression found" not in output
