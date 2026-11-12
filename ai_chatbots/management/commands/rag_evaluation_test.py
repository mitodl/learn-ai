"""Tests for the rag_evaluation command's flag rules."""

import pytest
from django.core.management import CommandError, call_command

# Imported at collection, as the evaluation tests do: the orchestrator uses
# deepeval's LLMTestCaseParams, which warns (an error inside a test) on import
from ai_chatbots.management.commands import rag_evaluation  # noqa: F401

pytestmark = pytest.mark.django_db


@pytest.fixture
def run_evaluation(mocker):
    """Keep the command from generating or judging anything."""
    return mocker.patch(
        "ai_chatbots.management.commands.rag_evaluation."
        "EvaluationOrchestrator.run_evaluation",
        new=mocker.AsyncMock(),
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ("--models a,b --baseline c", "exactly one other model"),
        (
            "--models a,b --baseline a --pairwise-only --prompts --bots syllabus",
            "one prompt per side",
        ),
        (
            "--models a,b --baseline a --candidate-prompt x",
            "need --baseline and one bot",
        ),
        ("--models a,b --pairwise-only --correctness-only", "can't be combined"),
        ("--models a --pairwise-only", "--pairwise-only needs two"),
    ],
)
def test_refused_flags(run_evaluation, args, message):
    """Flag combinations without a meaningful result are refused."""
    with pytest.raises(CommandError, match=message):
        call_command("rag_evaluation", *args.split())
    run_evaluation.assert_not_called()


@pytest.mark.parametrize(
    ("args", "sample"),
    [
        ("--baseline a --bots syllabus", 30),
        ("--baseline a --pairwise-only --bots syllabus --sample 0", 0),
        ("--pairwise", None),
        ("--baseline a --bots tutor", 30),
    ],
)
def test_defaults(run_evaluation, args, sample):
    """Comparisons sample 30 cases, other runs every case; the judge is gpt-5.5."""
    call_command("rag_evaluation", "--models", "a,b", *args.split())
    kwargs = run_evaluation.call_args.kwargs
    assert kwargs["sample"] == sample
    # A baseline run always judges pairwise
    assert kwargs["pairwise"] or kwargs["baseline"]
    assert "Model settings" in kwargs["run_info"]
    config = run_evaluation.call_args.args[0]
    assert config.evaluation_model == "gpt-5.5"
    assert ("TutoringQuality" in config.metric_thresholds) == ("tutor" in args)
    # A comparison with no metric flag judges pairwise only, tutor aside
    assert config.pairwise_only == ("syllabus" in args)


def test_baseline_joins_models(run_evaluation):
    """--models can name just the candidate."""
    call_command(
        "rag_evaluation", "--models", "b", "--baseline", "a", "--bots", "syllabus"
    )
    assert run_evaluation.call_args.args[0].models == ["a", "b"]
