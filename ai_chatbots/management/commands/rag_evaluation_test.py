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
        ("--models a,a --baseline a", "name the same model"),
        (
            "--models a --baseline a --bots syllabus "
            "--baseline-prompt x/p.txt --candidate-prompt y/p.txt",
            "name the same model",
        ),
        ("--models a,b --data-file x/c.json,y/c.json", "names must differ"),
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
        ("--baseline a --bots syllabus", None),
        ("--baseline a --pairwise-only --bots syllabus --sample 0", 0),
        ("--pairwise", None),
        ("--baseline a --bots tutor", None),
    ],
)
def test_defaults(run_evaluation, args, sample):
    """Every case runs unless --sample says; the judge is gpt-5.5."""
    call_command("rag_evaluation", "--models", "a,b", *args.split())
    kwargs = run_evaluation.call_args.kwargs
    assert kwargs["sample"] == sample
    # A baseline run always judges pairwise
    assert kwargs["pairwise"] or kwargs["baseline"]
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


def test_baseline_same_model_two_prompts(run_evaluation, tmp_path):
    """One model can be its own baseline when the sides' prompts differ."""
    for name in ("old", "new"):
        (tmp_path / f"{name}.txt").write_text(f"{name} prompt")
    call_command(
        "rag_evaluation",
        *f"--models a --baseline a --bots syllabus --baseline-prompt {tmp_path}/old.txt "
        f"--candidate-prompt {tmp_path}/new.txt".split(),
    )
    kwargs = run_evaluation.call_args.kwargs
    assert run_evaluation.call_args.args[0].models == ["a"]
    assert (kwargs["baseline_prompt"]["name"], kwargs["candidate_prompt"]["name"]) == (
        "old",
        "new",
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Be brief.", "needs {problem_statement}"),
        ('{"problem": "Be brief."}', "Unknown tutor prompts"),
    ],
)
def test_tutor_side_prompt_refused(run_evaluation, tmp_path, text, message):
    """A tutor prompt it can't use is refused before anything runs."""
    (tmp_path / "p.txt").write_text(text)
    with pytest.raises(CommandError, match=message):
        call_command(
            "rag_evaluation",
            *f"--models a,b --baseline a --bots tutor --candidate-prompt {tmp_path}/p.txt".split(),
        )
    run_evaluation.assert_not_called()
