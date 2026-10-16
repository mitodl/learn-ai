"""Tests for the account tools, which read the learner's own records"""

import json

import pytest

from ai_chatbots.learner_api import LearnerAPIError
from ai_chatbots.tools import (
    COULD_NOT_CHECK,
    get_certificate_status,
    get_course_progress,
    get_enrollments,
    get_upcoming_assignments,
)

COURSES_URL = "https://courses.learn.test/learn/course"


def enrollment(  # noqa: PLR0913
    course_id,
    run_tag,
    title,
    *,
    start="2020-01-01T00:00:00Z",
    end="2099-01-01T00:00:00Z",
    grade=None,
    certificate=None,
):
    """Build an enrollment shaped like MITx Online's /api/v3/enrollments/."""
    run_id = f"{course_id}+{run_tag}"
    return {
        "id": 1,
        "b2b_contract_id": None,
        "edx_emails_subscription": True,
        "enrollment_mode": "verified",
        "certificate": certificate,
        "grades": [grade] if grade else [],
        "run": {
            "courseware_id": run_id,
            "courseware_url": f"{COURSES_URL}/{run_id}/home",
            "title": title,
            "start_date": start,
            "end_date": end,
            "is_self_paced": False,
            "upgrade_product_price": "100.00",
            "course": {"readable_id": course_id, "title": title},
        },
    }


CALCULUS = "course-v1:MITxT+18.01x"
ECONOMICS = "course-v1:MITxT+14.750x"
ECONOMICS_2 = "course-v1:MITxT+14.73x"

ENROLLMENTS = [
    enrollment(
        CALCULUS,
        "1T2020",
        "Calculus 1A",
        end="2020-06-01T00:00:00Z",
        grade={
            "grade": 0.41,
            "grade_percent": 41,
            "letter_grade": "F",
            "passed": False,
        },
    ),
    enrollment(
        CALCULUS,
        "3T2026",
        "Calculus 1A",
        grade={
            "grade": 0.88,
            "grade_percent": 88,
            "letter_grade": "B",
            "passed": True,
        },
    ),
    enrollment(
        ECONOMICS,
        "3T2023",
        "Political Economy and Economic Development",
        end="2023-12-20T15:00:00Z",
        certificate={"link": "/certificate/abc/", "uuid": "abc"},
    ),
    enrollment(
        ECONOMICS_2,
        "1T2099",
        "Foundations of Development Policy",
        start="2099-01-01T00:00:00Z",
        end=None,
    ),
]


@pytest.fixture
def records(mocker):
    """Return the learner's records, as learner_api returns them."""
    api = "ai_chatbots.tools.learner_api"
    return {
        "enrollments": mocker.patch(f"{api}.get_enrollments", return_value=ENROLLMENTS),
        "programs": mocker.patch(
            f"{api}.get_program_enrollments",
            return_value=[
                {
                    "program": {"title": "Data, Economics", "readable_id": "DEDP"},
                    "certificate": None,
                    "enrollment_mode": "verified",
                }
            ],
        ),
        "program_certificates": mocker.patch(
            f"{api}.get_program_certificates",
            return_value=[
                {
                    "program_title": "Data, Economics",
                    "program_completion_timestamp": "2025-01-01T00:00:00Z",
                    "program_letter_share_url": "https://learn.test/letter/1",
                    "user_street_address": "1 Main St",
                    "user_year_of_birth": 1990,
                    "user_gender": "f",
                    "user_email": "learner@example.com",
                }
            ],
        ),
    }


async def call(tool, course=None, page_course=None, platform=None):
    """Invoke a tool the way the agent does, and decode its JSON."""
    state = {"course_id": [page_course]} if page_course else {}
    if platform:
        state["platform"] = [platform]
    return json.loads(await tool.ainvoke({"course": course, "state": state}))


def test_model_sees_only_the_course_argument():
    """The agent state is injected, never part of what the model can fill in."""
    for tool in (
        get_enrollments,
        get_course_progress,
        get_certificate_status,
        get_upcoming_assignments,
    ):
        properties = tool.tool_call_schema.model_json_schema()["properties"]
        assert list(properties) == ["course"]


async def test_progress_defaults_to_the_page_course(records):
    """Without a course, the course of the page is used, at its current run."""
    result = await call(get_course_progress, page_course=CALCULUS)

    assert result["course_on_page"] == CALCULUS
    assert result["enrollments"] == [
        {
            "course_title": "Calculus 1A",
            "course_id": CALCULUS,
            "run_id": f"{CALCULUS}+3T2026",
            "status": "in progress",
            "grade": {"percent": 88, "letter": "B", "passed": True},
            "progress_url": f"{COURSES_URL}/{CALCULUS}+3T2026/progress",
        }
    ]


@pytest.mark.parametrize(
    ("course", "run_id"),
    [
        ("14.750x", f"{ECONOMICS}+3T2023"),
        ("political economy", f"{ECONOMICS}+3T2023"),
        ("18.01", f"{CALCULUS}+3T2026"),
        (f"{ECONOMICS}+3T2023", f"{ECONOMICS}+3T2023"),
        ("18.01x", f"{CALCULUS}+3T2026"),
    ],
)
async def test_named_course_overrides_the_page(records, course, run_id):
    """A course named by number, title or id is used instead of the page's."""
    result = await call(get_course_progress, course=course, page_course=CALCULUS)

    assert [e["run_id"] for e in result["enrollments"]] == [run_id]


async def test_ambiguous_course_asks_which(records):
    """A name matching several courses returns them all, to ask which."""
    result = await call(get_course_progress, course="Development")

    assert {e["course_id"] for e in result["enrollments"]} == {ECONOMICS, ECONOMICS_2}
    assert "ask which" in result["note"]


async def test_unmatched_course_says_no_enrollment(records):
    """A course the learner isn't enrolled in is reported as such."""
    result = await call(get_course_progress, course="8.01x")

    assert result["enrollments"] == []
    assert "no MITx Online enrollment matching" in result["note"]


async def test_page_course_not_enrolled_lists_all(records):
    """On the page of a course the learner isn't in, list what they are in."""
    result = await call(
        get_course_progress, page_course="course-v1:MITxT+8.01x", platform="mitxonline"
    )

    assert "no MITx Online enrollment in the course on this page" in result["note"]
    assert len(result["enrollments"]) == 3


async def test_page_on_another_platform_lists_nothing(records):
    """On a course from another platform, say it can't be checked here."""
    result = await call(get_course_progress, page_course="MITx+6.00.1x", platform="edx")

    assert result["enrollments"] == []
    assert "isn't on MITx Online" in result["note"]


async def test_earlier_certificate_isnt_hidden_by_a_newer_run(records):
    """A certificate from an ended run shows even with a newer run enrolled."""
    records["enrollments"].return_value = [
        enrollment(
            ECONOMICS,
            "3T2023",
            "Political Economy",
            end="2023-12-20T15:00:00Z",
            certificate={"link": "/certificate/abc/"},
        ),
        enrollment(ECONOMICS, "1T2099", "Political Economy", start="2099-01-01Z"),
    ]

    certificate = await call(get_certificate_status, course="14.750x")
    progress = await call(get_course_progress, course="14.750x")

    assert certificate["enrollments"][0]["certificate_earned"] is True
    assert progress["enrollments"][0]["run_id"] == f"{ECONOMICS}+1T2099"


async def test_enrollments_list_all_courses_on_a_course_page(records):
    """Asking what the learner is enrolled in lists every course, not the page's."""
    result = await call(get_enrollments, page_course=CALCULUS)

    assert {e["course_id"]: e["status"] for e in result["enrollments"]} == {
        CALCULUS: "in progress",
        ECONOMICS: "ended",
        ECONOMICS_2: "upcoming",
    }
    assert "grade" not in result["enrollments"][0]


async def test_enrollments_keep_only_answer_fields(records):
    """Payment and email settings never reach the model."""
    result = await call(get_enrollments)

    text = json.dumps(result)
    for field in ("upgrade_product_price", "edx_emails_subscription", "b2b"):
        assert field not in text


async def test_unrecorded_grade_is_not_zero(records):
    """MITx Online's 0 grade_percent with no grade recorded isn't reported as 0%."""
    records["enrollments"].return_value = [
        enrollment(
            CALCULUS,
            "3T2026",
            "Calculus 1A",
            grade={"grade": None, "grade_percent": 0, "passed": False},
        )
    ]

    result = await call(get_course_progress, page_course=CALCULUS)

    assert result["enrollments"][0]["grade"] is None


async def test_upcoming_assignments_link_the_dates_tab(records):
    """Due dates are a link to the course's Dates tab."""
    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    assert result["enrollments"][0]["dates_url"] == (
        f"{COURSES_URL}/{CALCULUS}+3T2026/dates"
    )


async def test_certificate_status_drops_personal_data(settings, records):
    """Program certificates keep their title, date and link, nothing personal."""
    settings.LEARNER_API_MITXONLINE_BASE_URL = "https://mitxonline.test/"
    result = await call(get_certificate_status, course="14.750x")

    assert result["enrollments"][0]["certificate_earned"] is True
    assert result["enrollments"][0]["certificate_url"] == (
        "https://mitxonline.test/certificate/abc/"
    )
    assert result["program_certificates"] == [
        {
            "program_title": "Data, Economics",
            "completed_on": "2025-01-01T00:00:00Z",
            "certificate_url": "https://learn.test/letter/1",
        }
    ]
    assert result["programs"][0]["certificate_earned"] is False
    text = json.dumps(result)
    for value in ("1 Main St", "1990", "learner@example.com"):
        assert value not in text


@pytest.mark.parametrize(
    ("tool", "failing"),
    [
        (get_enrollments, "enrollments"),
        (get_course_progress, "enrollments"),
        (get_upcoming_assignments, "enrollments"),
        (get_certificate_status, "enrollments"),
        (get_certificate_status, "programs"),
        (get_certificate_status, "program_certificates"),
    ],
)
async def test_failed_lookup_says_could_not_check(records, tool, failing):
    """A failed lookup is "couldn't check", never an empty answer."""
    records[failing].side_effect = LearnerAPIError(failing, 403)

    result = await call(tool, page_course=CALCULUS)

    assert result == {"error": COULD_NOT_CHECK}
