"""Compare a candidate's answers with the baseline's, case by case."""

from math import comb

SIGNIFICANCE = 0.05
CORRECTNESS_THRESHOLD = 0.7
# The fewest differing cases that can ever be flagged: 6 worse, none better
MIN_FLAGGABLE = 6
BETTER, WORSE, SAME = "better", "worse", "same"


def sign_test_p(worse: int, better: int) -> float:
    """Exact two-sided sign test on the cases where the two models differ."""
    n = worse + better
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(max(worse, better), n + 1)) / 2**n
    return min(1.0, 2 * tail)


def _outcome(baseline_failed, candidate_failed) -> str | None:
    """Compare pass/fail on one case; None when either side wasn't judged."""
    if baseline_failed is None or candidate_failed is None:
        return None
    if candidate_failed != baseline_failed:
        return WORSE if candidate_failed else BETTER
    return SAME


def measure_outcomes(  # noqa: PLR0913
    records: list[dict],
    scores: dict[tuple, float],
    winners: dict[str, str | None],
    baseline: str,
    candidate: str,
    skip: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """
    Give each measure's outcome for the candidate (better, worse or same) on
    every case both models answered, leaving out the cases in skip.

    records are the answers' records (case_id, model, question, checks), scores
    the judged Correctness scores by (case_id, model), and winners the pairwise
    winner per case.  Returns {measure: {case_id: outcome}} and the questions.
    """
    by_case: dict[str, dict[str, dict]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], {})[record["model"]] = record
    cases = [
        case_id
        for case_id, by_model in by_case.items()
        if case_id not in skip and {baseline, candidate} <= set(by_model)
    ]

    def below(case_id, model):
        score = scores.get((case_id, model))
        return None if score is None else score < CORRECTNESS_THRESHOLD

    def failed(name):
        return lambda case_id, model: bool(by_case[case_id][model]["checks"].get(name))

    measures = {
        "pairwise": lambda case_id, model: (
            None if winners.get(case_id) is None else winners[case_id] != model
        )
    }
    if scores:
        measures["Correctness"] = below
    checks = {
        name for case_id in cases for name in by_case[case_id][baseline]["checks"]
    }
    measures.update({name: failed(name) for name in sorted(checks)})

    outcomes = {
        name: {
            case_id: outcome
            for case_id in cases
            if (
                outcome := _outcome(fails(case_id, baseline), fails(case_id, candidate))
            )
        }
        for name, fails in measures.items()
    }
    return outcomes, {c: by_case[c][baseline]["question"] for c in cases}


def print_comparison(
    write,
    outcomes: dict[str, dict[str, str]],
    questions: dict[str, str],
    *,
    failed: bool,
) -> list[str]:
    """
    Print each measure's better/worse/same counts and whether the candidate is
    clearly worse on it: worse more often than better, with sign test p below
    SIGNIFICANCE.  Returns the measures flagged.
    """
    flagged = []
    for name, by_case in outcomes.items():
        results = list(by_case.values())
        worse, better = results.count(WORSE), results.count(BETTER)
        p_value = sign_test_p(worse, better)
        if worse + better < MIN_FLAGGABLE:
            flag = "too few cases to flag"
        elif worse > better and p_value < SIGNIFICANCE:
            flag = "🚩 CLEARLY WORSE"
            flagged.append(name)
        else:
            flag = "not flagged"
        write(
            f"  {name}: better {better}, worse {worse}, same {results.count(SAME)}; "
            f"p={p_value:.3f}; {flag}"
        )
        # The cases to read: a flagged code check's, and every Correctness loss
        if name != "pairwise" and (name in flagged or name == "Correctness"):
            for case_id, outcome in by_case.items():
                if outcome == WORSE:
                    write(f"     ↳ worse: {case_id}: {questions.get(case_id, '')[:80]}")
    if failed or not questions:
        why = "see FAILURES" if failed else "no cases compared"
        write(f"  Incomplete ({why}): rerun before reading anything into this.")
    elif flagged:
        write(f"  Regression found: clearly worse on {', '.join(flagged)}.")
    else:
        write(
            f"  No regression found on {len(questions)} cases. "
            "Small samples miss small losses."
        )
    return flagged
