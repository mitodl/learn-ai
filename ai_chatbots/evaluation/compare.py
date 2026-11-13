"""Compare a candidate's answers with the baseline's, case by case."""

from math import comb

SIGNIFICANCE = 0.05
JUDGED_SCORES = ("Correctness", "TutoringQuality")
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
    thresholds: dict[str, float],
    winners: dict[str, str | None],
    baseline: str,
    candidate: str,
    skip: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Give the candidate's outcome per measure and case both sides answered."""
    by_case: dict[str, dict[str, dict]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], {})[record["side"]] = record
    cases = [
        case_id
        for case_id, by_side in by_case.items()
        if case_id not in skip and {baseline, candidate} <= set(by_side)
    ]

    def fails(case_id, side) -> dict[str, bool | None]:
        """Whether one answer failed each measure; None when it wasn't judged."""
        record, winner = by_case[case_id][side], winners.get(case_id)
        scores = {m: record["scores"].get(m) for m in thresholds}
        return {
            "pairwise": None if winner is None else winner != side,
            **{m: None if s is None else s < thresholds[m] for m, s in scores.items()},
            **{name: bool(record["checks"][name]) for name in sorted(record["checks"])},
        }

    outcomes: dict[str, dict[str, str]] = {}
    for case_id in cases:
        candidate_fails = fails(case_id, candidate)
        for name, baseline_failed in fails(case_id, baseline).items():
            outcome = _outcome(baseline_failed, candidate_fails.get(name))
            outcomes.setdefault(name, {})
            if outcome:
                outcomes[name][case_id] = outcome
    return outcomes, {c: by_case[c][baseline]["question"] for c in cases}


def print_comparison(
    write,
    outcomes: dict[str, dict[str, str]],
    questions: dict[str, str],
    *,
    failed: bool,
) -> list[str]:
    """Print each measure's counts and flag; return the measures flagged."""
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
        # The cases to read: a flagged code check's, and every judged score loss
        if name != "pairwise" and (name in flagged or name in JUDGED_SCORES):
            for case_id, outcome in by_case.items():
                if outcome == WORSE:
                    write(f"     ↳ worse: {case_id}: {questions.get(case_id, '')[:80]}")
    # A measure with nothing compared (say, no score recorded) shows nothing
    unmeasured = ", ".join(name for name, by_case in outcomes.items() if not by_case)
    if failed or not questions or unmeasured:
        on = unmeasured and f" on {unmeasured}"
        why = "see FAILURES" if failed else f"no cases compared{on}"
        so_far = f" Clearly worse so far on {', '.join(flagged)}." if flagged else ""
        write(f"  Incomplete ({why}): rerun before relying on this.{so_far}")
    elif flagged:
        write(f"  Regression found: clearly worse on {', '.join(flagged)}.")
    else:
        write(
            f"  Nothing flagged on {len(questions)} cases. "
            "That doesn't rule out a regression."
        )
    return flagged
