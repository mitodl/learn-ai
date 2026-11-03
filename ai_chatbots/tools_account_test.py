"""Tests for the account tools, which read the learner's own records"""

import json

import pytest
from django.core.cache import caches

from ai_chatbots.learner_api import LearnerAPIError
from ai_chatbots.tools import (
    COULD_NOT_CHECK,
    MAX_DUE_DATES,
    MITXONLINE_PLATFORM,
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
# A readable id MITx Online and xPRO share
SHARED = "course-v1:PRO+AIGE"
PROGRAM = "program-v1:MITx+DEDP"

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


# Shaped like MITx Online's /api/v2/course_certificates/<uuid>/
ISSUED_CERTIFICATE = {
    "uuid": "abc",
    "is_revoked": False,
    "user": {"name": "Jane Learner"},
    "verifiable_credential_json": {"credentialSubject": {"name": "Jane Learner"}},
}


@pytest.fixture
def records(mocker, settings):
    """Return the learner's records, as learner_api returns them."""
    settings.LEARNER_API_OPENEDX_BASE_URL = "https://courses.learn.test/"
    settings.LEARNER_API_OPENEDX_COOKIE_NAME = "edx-session"
    api = "ai_chatbots.tools.learner_api"
    mocker.patch(
        "ai_chatbots.tools._get_course_platform", return_value=MITXONLINE_PLATFORM
    )
    return {
        "enrollments": mocker.patch(f"{api}.get_enrollments", return_value=ENROLLMENTS),
        "programs": mocker.patch(
            f"{api}.get_program_enrollments",
            return_value=[
                {
                    "program": {"title": "Data, Economics", "readable_id": PROGRAM},
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
        "course_certificate": mocker.patch(
            f"{api}.get_course_certificate", return_value=ISSUED_CERTIFICATE
        ),
        "program_certificate": mocker.patch(
            f"{api}.get_program_certificate", return_value=ISSUED_CERTIFICATE
        ),
        "course_dates": mocker.patch(
            f"{api}.get_course_dates", return_value={"course_date_blocks": []}
        ),
    }


async def call(tool, page_course=None, platform=None, course_id=None, **args):
    """Invoke a tool the way the agent does, and decode its JSON."""
    state = {"course_id": [page_course]} if page_course else {}
    if platform:
        state["platform"] = [platform]
    args["state"] = state
    if course_id:
        args["course_id"] = course_id
    return json.loads(await tool.ainvoke(args))


def test_model_fills_in_only_the_course():
    """The agent state is injected, never part of what the model can fill in."""
    for tool, fields in (
        (get_enrollments, ["course_id"]),
        (get_course_progress, ["course_id"]),
        (get_certificate_status, ["course_id", "all_courses"]),
        (get_upcoming_assignments, ["course_id"]),
    ):
        schema = tool.tool_call_schema.model_json_schema()
        assert list(schema["properties"]) == fields
        assert not schema.get("required")


async def test_progress_is_for_the_page_course(records):
    """The course of the page is used, at its current run."""
    result = await call(get_course_progress, page_course=CALCULUS)

    assert result["course_on_page"] == CALCULUS
    assert result["completion"].startswith("Not available")
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


async def test_page_run_id_matches_that_run(records):
    """A page with a run id gets that run, not the course's current one."""
    result = await call(get_course_progress, page_course=f"{CALCULUS}+1T2020")

    assert [e["run_id"] for e in result["enrollments"]] == [f"{CALCULUS}+1T2020"]


@pytest.mark.parametrize(
    "page_course",
    [
        "course-v1:MITxT+8.01x",
        # Same characters as 18.01x without the dots
        "course-v1:MITxT+1.801x",
        "abc.123x",
        "MITxT",
    ],
)
async def test_page_course_not_enrolled_lists_nothing(records, page_course):
    """On a course the learner isn't in, no other course is offered instead."""
    result = await call(get_course_progress, page_course=page_course)

    assert result["enrollments"] == []
    assert "no MITx Online enrollment in the course on this page" in result["note"]


@pytest.mark.parametrize(
    "tool", [get_course_progress, get_certificate_status, get_upcoming_assignments]
)
async def test_no_page_course_asks_for_one(records, tool):
    """Without a course on the page or a course_id, no records are read."""
    result = await call(tool)

    assert result["enrollments"] == []
    assert "programs" not in result
    assert "Pass the course the learner asked about" in result["note"]
    for api in records.values():
        api.assert_not_called()


@pytest.mark.parametrize("page_course", [None, CALCULUS])
@pytest.mark.parametrize(
    ("course_id", "run_id"),
    [
        (ECONOMICS_2, f"{ECONOMICS_2}+1T2099"),
        (f"{ECONOMICS}+3T2023", f"{ECONOMICS}+3T2023"),
    ],
)
async def test_course_id_reads_that_course(
    mocker, records, page_course, course_id, run_id
):
    """A course_id from get_enrollments reads that course, with no platform lookup."""
    lookup = mocker.patch("ai_chatbots.tools._get_course_platform")

    result = await call(
        get_course_progress, page_course=page_course, course_id=course_id
    )

    assert [e["run_id"] for e in result["enrollments"]] == [run_id]
    assert "course_on_page" not in result
    lookup.assert_not_called()


@pytest.mark.parametrize("page_course", [None, CALCULUS])
@pytest.mark.parametrize(
    "course_id",
    [
        "course-v1:MITxT+8.01x",
        "course-v1:MITxT+1.801x",
        "abc.123x",
        "MITxT",
        # A run tag ends many run ids, so it names none
        "3T2026",
    ],
)
async def test_course_id_not_enrolled_lists_nothing(records, page_course, course_id):
    """An id the learner isn't enrolled in gets nothing, not another course."""
    result = await call(
        get_course_progress, page_course=page_course, course_id=course_id
    )

    assert result["enrollments"] == []
    assert f"{course_id} isn't in the learner's MITx Online records" in result["note"]
    assert "get_enrollments" in result["note"]


@pytest.mark.parametrize(
    ("page_course", "course_id", "runs"),
    [
        # Numbers and ids: the end of a course id after ":" or "+"
        (ECONOMICS, "18.01x", [f"{CALCULUS}+3T2026"]),
        (ECONOMICS, "18.01X", [f"{CALCULUS}+3T2026"]),
        (ECONOMICS, "MITxT+18.01x", [f"{CALCULUS}+3T2026"]),
        (ECONOMICS, "8.01x", ["course-v1:8.01x+1T2026"]),
        # Without the trailing "x"
        (ECONOMICS, "18.01", [f"{CALCULUS}+3T2026"]),
        (ECONOMICS, "8.01", ["course-v1:8.01x+1T2026"]),
        (ECONOMICS, "1.801", []),
        # Titles: whole words only
        (ECONOMICS, "Foundations of Development Policy", [f"{ECONOMICS_2}+1T2099"]),
        (ECONOMICS, "foundations of  development policy", [f"{ECONOMICS_2}+1T2099"]),
        (ECONOMICS, "Calculus 1A", [f"{CALCULUS}+3T2026"]),
        (ECONOMICS, "Calculus 1", []),
        (ECONOMICS, "Develop", []),
        # On an older run's page, its course by any name reads that run
        (f"{CALCULUS}+1T2020", None, [f"{CALCULUS}+1T2020"]),
        (f"{CALCULUS}+1T2020", "Calculus 1A", [f"{CALCULUS}+1T2020"]),
        (f"{CALCULUS}+1T2020", "18.01x", [f"{CALCULUS}+1T2020"]),
        # A run asked for by its id keeps that run
        (f"{CALCULUS}+1T2020", f"{CALCULUS}+3T2026", [f"{CALCULUS}+3T2026"]),
        (CALCULUS, f"{CALCULUS}+1T2020", [f"{CALCULUS}+1T2020"]),
    ],
)
async def test_named_course_reads_its_runs(records, page_course, course_id, runs):
    """A course named by number, id or title reads that course's run."""
    records["enrollments"].return_value = [
        *ENROLLMENTS,
        enrollment("course-v1:8.01x", "1T2026", "Physics I"),
    ]

    result = await call(
        get_course_progress, page_course=page_course, course_id=course_id
    )

    assert [e["run_id"] for e in result["enrollments"]] == runs


async def test_page_course_title_of_another_run_reads_the_page_run(records):
    """A title only an older run has still reads the page's run of that course."""
    runs = [
        enrollment(CALCULUS, "1T2020", "Calculus 1A 2020"),
        enrollment(CALCULUS, "3T2026", "Differentiation"),
    ]
    for run in runs:
        # Without a course title, each run's own title is used
        run["run"]["course"]["title"] = None
    records["enrollments"].return_value = runs

    result = await call(
        get_course_progress, page_course=f"{CALCULUS}+3T2026", course_id="Calculus 1A"
    )

    assert [e["run_id"] for e in result["enrollments"]] == [f"{CALCULUS}+3T2026"]


@pytest.mark.parametrize("page_course", [None, CALCULUS])
@pytest.mark.parametrize(
    ("enrolled", "course_id", "named"),
    [
        (["course-v1:MITxT+AIGE", SHARED], "AIGE", f"course-v1:MITxT+AIGE, {SHARED}"),
        ([SHARED, "course-v1:MITxT+AIGE"], "AIGE", f"course-v1:MITxT+AIGE, {SHARED}"),
        # Words two titles share
        (None, "development", f"{ECONOMICS_2}, {ECONOMICS}"),
    ],
)
async def test_name_of_two_courses_asks_which(
    records, page_course, enrolled, course_id, named
):
    """A name two of the learner's courses share gets no records, only both ids."""
    if enrolled:
        records["enrollments"].return_value = [
            enrollment(course, "1T2026", course) for course in enrolled
        ]

    result = await call(
        get_course_progress, page_course=page_course, course_id=course_id
    )

    assert result["enrollments"] == []
    assert named in result["note"]
    assert "Ask which one they mean" in result["note"]


@pytest.mark.parametrize("tool", [get_course_progress, get_enrollments])
@pytest.mark.parametrize("page_course", [SHARED, f"{SHARED}+1T2020"])
@pytest.mark.parametrize("platform", ["xpro", None])
async def test_course_number_of_the_page_course_is_the_page_course(
    mocker, records, tool, page_course, platform
):
    """
    The page course's number reads the page's course, not a MITx Online course
    from another organization with the same number. A run page's course is
    known from the learner's enrollment in that run.
    """
    mocker.patch("ai_chatbots.tools._get_course_platform", return_value="xpro")
    records["enrollments"].return_value = [
        enrollment("course-v1:OTHER+AIGE", "1T2026", "AI Governance"),
        enrollment(SHARED, "1T2020", "AI Governance"),
    ]

    result = await call(
        tool, page_course=page_course, platform=platform, course_id="AIGE"
    )

    assert result["enrollments"] == []


async def test_run_tag_on_a_run_page_names_nothing(records):
    """A bare run tag isn't read as the run page's course."""
    result = await call(
        get_course_progress, page_course=f"{CALCULUS}+1T2020", course_id="1T2020"
    )

    assert result["enrollments"] == []
    assert "1T2020 isn't in the learner's MITx Online records" in result["note"]


@pytest.mark.parametrize(
    ("page_course", "course_id", "enrolled"),
    [
        # The page's own id
        (SHARED, SHARED, [(SHARED, "1T2026")]),
        # The page course's number
        (SHARED, "AIGE", [(SHARED, "1T2026")]),
        # A run of a course page
        (SHARED, f"{SHARED}+1T2026", [(SHARED, "1T2026")]),
        # The course of a run page, whose current run isn't the page's
        (f"{SHARED}+1T2020", SHARED, [(SHARED, "1T2020"), (SHARED, "1T2026")]),
        # A sibling run of a run page
        (
            f"{SHARED}+1T2020",
            f"{SHARED}+1T2026",
            [(SHARED, "1T2020"), (SHARED, "1T2026")],
        ),
    ],
)
@pytest.mark.parametrize("platform", ["xpro", None])
@pytest.mark.parametrize("tool", [get_course_progress, get_enrollments])
async def test_page_course_by_any_name_keeps_its_platform_check(  # noqa: PLR0913
    mocker, records, page_course, course_id, enrolled, platform, tool
):
    """
    A course_id naming the page's course, under any of its ids or its title,
    is still checked against the page's platform: the page could be xPRO's
    course with the same id as the learner's MITx Online one.
    """
    mocker.patch("ai_chatbots.tools._get_course_platform", return_value="xpro")
    records["enrollments"].return_value = [
        enrollment(course, run, "Calculus 1A", start=f"20{run[-2:]}-01-01T00:00:00Z")
        for course, run in enrolled
    ]

    result = await call(
        tool, page_course=page_course, platform=platform, course_id=course_id
    )

    assert result["enrollments"] == []
    assert "isn't on MITx Online" in result["note"]


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
        enrollment(
            ECONOMICS,
            "1T2099",
            "Political Economy",
            start="2099-01-01T00:00:00Z",
            end=None,
        ),
    ]

    certificate = await call(get_certificate_status, page_course=ECONOMICS)
    progress = await call(get_course_progress, page_course=ECONOMICS)

    assert certificate["enrollments"][0]["certificate_earned"] is True
    assert progress["enrollments"][0]["run_id"] == f"{ECONOMICS}+1T2099"
    certificate = await call(
        get_certificate_status, page_course=CALCULUS, course_id=ECONOMICS
    )
    assert certificate["enrollments"][0]["certificate_earned"] is True


async def test_latest_run_compares_dates_not_text(records):
    """The latest of two ended runs is picked by time, not by the date's text."""
    records["enrollments"].return_value = [
        # 10:00+05:00 is 05:00 UTC, earlier than the other run's 06:00 UTC
        enrollment(
            CALCULUS,
            "earlier",
            "Calculus 1A",
            start="2020-03-01T10:00:00+05:00",
            end="2020-06-01T00:00:00Z",
        ),
        enrollment(
            CALCULUS,
            "later",
            "Calculus 1A",
            start="2020-03-01T06:00:00Z",
            end="2020-06-01T00:00:00Z",
        ),
    ]

    result = await call(get_course_progress, page_course=CALCULUS)

    assert result["enrollments"][0]["run_id"] == f"{CALCULUS}+later"


async def test_enrollments_list_all_courses_on_a_course_page(records):
    """Asking what the learner is enrolled in lists every course, not the page's."""
    result = await call(get_enrollments, page_course=CALCULUS)

    assert {e["course_id"]: e["status"] for e in result["enrollments"]} == {
        CALCULUS: "in progress",
        ECONOMICS: "ended",
        ECONOMICS_2: "upcoming",
    }
    assert result["course_on_page"].startswith(
        "The learner is enrolled in the course on this page, Calculus 1A "
        f"({CALCULUS}); its run is in progress."
    )
    assert "grade" not in result["enrollments"][0]


@pytest.mark.parametrize(
    ("course_id", "runs"),
    [
        ("18.01x", [f"{CALCULUS}+3T2026"]),
        # An older run asked for by its id isn't hidden by the current one
        (f"{CALCULUS}+1T2020", [f"{CALCULUS}+1T2020"]),
        ("Calculus 1A", [f"{CALCULUS}+3T2026"]),
        ("8.01x", []),
    ],
)
async def test_enrollments_for_a_named_course(records, course_id, runs):
    """With course_id, only the course it names is listed."""
    result = await call(get_enrollments, page_course=ECONOMICS, course_id=course_id)

    assert [e["run_id"] for e in result["enrollments"]] == runs
    assert ("isn't in the learner's MITx Online records" in result.get("note", "")) is (
        not runs
    )


@pytest.mark.parametrize("platform", ["xpro", None])
async def test_enrollments_dont_name_another_platforms_page_course(
    mocker, records, platform
):
    """A MITx Online course sharing an xPRO page's id isn't called the page's."""
    mocker.patch("ai_chatbots.tools._get_course_platform", return_value="xpro")
    records["enrollments"].return_value = [enrollment(SHARED, "1T2026", "AI for Good")]

    result = await call(get_enrollments, page_course=SHARED, platform=platform)

    assert [e["course_id"] for e in result["enrollments"]] == [SHARED]
    assert "isn't on MITx Online" in result["course_on_page"]


@pytest.mark.parametrize(
    ("enrolled", "runs"),
    [
        (["course-v1:OTHER+AIGE"], []),
        (["course-v1:OTHER+AIGE", SHARED], [f"{SHARED}+1T2026"]),
    ],
)
async def test_enrollments_by_the_page_course_name_on_mitxonline(
    records, enrolled, runs
):
    """On a MITx Online page, its number names it, not another course's."""
    records["enrollments"].return_value = [
        enrollment(course, "1T2026", "AI") for course in enrolled
    ]

    result = await call(
        get_enrollments,
        page_course=SHARED,
        platform=MITXONLINE_PLATFORM,
        course_id="AIGE",
    )

    assert [e["run_id"] for e in result["enrollments"]] == runs


@pytest.mark.parametrize("course_id", ["Physics I", "course-v1:8.01x+1T2026"])
async def test_enrollments_check_the_platform_of_the_run_pages_course(
    mocker, records, course_id
):
    """An unusual run id is checked by its course's platform, from the records."""
    lookup = mocker.patch(
        "ai_chatbots.tools._get_course_platform", return_value=MITXONLINE_PLATFORM
    )
    records["enrollments"].return_value = [
        enrollment("course-v1:8.01x", "1T2026", "Physics I")
    ]

    result = await call(
        get_enrollments, page_course="course-v1:8.01x+1T2026", course_id=course_id
    )

    assert [e["run_id"] for e in result["enrollments"]] == ["course-v1:8.01x+1T2026"]
    lookup.assert_called_with("course-v1:8.01x")


# The page's ended run counts on a run page too, though the list keeps the
# course's current run
@pytest.mark.parametrize("page_course", [ECONOMICS, f"{CALCULUS}+1T2020"])
async def test_enrollments_say_an_ended_page_course_is_still_enrolled(
    records, page_course
):
    """An ended run of the page's course is still an enrollment in it."""
    result = await call(get_enrollments, page_course=page_course)

    assert f"{CALCULUS}+3T2026" in [e["run_id"] for e in result["enrollments"]]
    assert result["course_on_page"].startswith(
        "The learner is enrolled in the course on this page"
    )
    assert "its run is ended" in result["course_on_page"]


async def test_enrollments_accept_dates_without_a_timezone(records):
    """A run date with no UTC offset still gets a status."""
    records["enrollments"].return_value = [
        enrollment(CALCULUS, "3T2026", "Calculus 1A", start="2099-01-01")
    ]

    result = await call(get_enrollments)

    assert result["enrollments"][0]["status"] == "upcoming"


async def test_enrollments_list_certificates(settings, records):
    """Each course and program says whether the learner earned its certificate."""
    settings.LEARNER_API_MITXONLINE_BASE_URL = "https://mitxonline.test/"
    result = await call(get_enrollments)

    assert {e["course_id"]: e["certificate"] for e in result["enrollments"]} == {
        CALCULUS: None,
        ECONOMICS: {
            "run_id": f"{ECONOMICS}+3T2023",
            "url": "https://mitxonline.test/certificate/abc/",
        },
        ECONOMICS_2: None,
    }
    assert result["programs"] == [
        {
            "program_title": "Data, Economics",
            "program_id": PROGRAM,
            "certificate_earned": False,
            "certificate_url": None,
        }
    ]


async def test_enrollments_keep_an_earlier_runs_certificate(records):
    """A retaken course lists the current run, and the earlier run's certificate."""
    records["enrollments"].return_value = [
        enrollment(
            CALCULUS,
            "1T2020",
            "Calculus 1A",
            end="2020-06-01T00:00:00Z",
            certificate={"link": "/certificate/abc/", "uuid": "abc"},
        ),
        enrollment(CALCULUS, "3T2026", "Calculus 1A"),
    ]

    (course,) = (await call(get_enrollments))["enrollments"]

    assert course["run_id"] == f"{CALCULUS}+3T2026"
    assert course["status"] == "in progress"
    assert course["end_date"] == "2099-01-01T00:00:00Z"
    assert course["certificate"]["run_id"] == f"{CALCULUS}+1T2020"


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
    assert "null means none is recorded yet" in result["grade_meaning"]


def due_date(title, date, date_type="assignment-due-date"):
    """Build a date block shaped like Open edX's /api/course_home/v1/dates/."""
    return {
        "assignment_type": "Homework",
        "complete": False,
        "date": date,
        "date_type": date_type,
        "description": "",
        "learner_has_access": True,
        "link": f"https://courses.learn.test/jump_to/{title}",
        "link_text": "",
        "title": title,
        "extra_info": None,
        "first_component_block_id": "block",
    }


async def test_upcoming_assignments_link_the_dates_tab(records):
    """The run's Dates tab is linked, with or without due dates."""
    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    assert result["enrollments"][0]["dates_url"] == (
        f"{COURSES_URL}/{CALCULUS}+3T2026/dates"
    )
    assert result["enrollments"][0]["upcoming_due_dates"] == []


async def test_upcoming_assignments_list_future_due_dates(records):
    """Only future assignment due dates are listed, soonest first."""
    records["course_dates"].return_value = {
        "course_date_blocks": [
            due_date("Course starts", "2020-01-01T00:00:00Z", "course-start-date"),
            due_date("Homework 1", "2020-02-01T00:00:00Z"),
            due_date("Homework 3", "2098-03-01T00:00:00Z"),
            due_date("Homework 2", "2098-02-01T00:00:00Z"),
            due_date("Course ends", "2099-01-01T00:00:00Z", "course-end-date"),
            due_date("No date", None),
        ]
    }

    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    records["course_dates"].assert_called_once_with(f"{CALCULUS}+3T2026")
    (summary,) = result["enrollments"]
    assert [due["title"] for due in summary["upcoming_due_dates"]] == [
        "Homework 2",
        "Homework 3",
    ]
    assert summary["upcoming_due_dates"][0] == {
        "title": "Homework 2",
        "assignment_type": "Homework",
        "due": "2098-02-01T00:00:00+00:00",
        "complete": False,
        "learner_has_access": True,
        "link": "https://courses.learn.test/jump_to/Homework 2",
    }
    assert summary["more_due_dates"] == 0


async def test_upcoming_assignments_are_capped(records):
    """Only the next MAX_DUE_DATES are listed; the rest are counted."""
    records["course_dates"].return_value = {
        "course_date_blocks": [
            due_date(f"Homework {day}", f"2098-01-{day:02d}T00:00:00Z")
            for day in range(1, MAX_DUE_DATES + 4)
        ]
    }

    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    (summary,) = result["enrollments"]
    assert len(summary["upcoming_due_dates"]) == MAX_DUE_DATES
    assert summary["upcoming_due_dates"][0]["title"] == "Homework 1"
    assert summary["more_due_dates"] == 3


async def test_upcoming_assignments_without_due_dates(records):
    """A failed due dates lookup still returns the run's dates and Dates tab."""
    records["course_dates"].side_effect = LearnerAPIError("get_course_dates")

    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    (summary,) = result["enrollments"]
    assert summary["dates_url"] == f"{COURSES_URL}/{CALCULUS}+3T2026/dates"
    assert summary["due_dates_note"] == "The due dates couldn't be checked."
    assert "upcoming_due_dates" not in summary


@pytest.mark.parametrize(
    "setting", ["LEARNER_API_OPENEDX_BASE_URL", "LEARNER_API_OPENEDX_COOKIE_NAME"]
)
async def test_upcoming_assignments_without_openedx_settings(
    settings, records, setting
):
    """Without the Open edX settings, only the Dates tab is linked, as before."""
    setattr(settings, setting, "")

    result = await call(get_upcoming_assignments, page_course=CALCULUS)

    assert result["enrollments"][0] == {
        "course_title": "Calculus 1A",
        "course_id": CALCULUS,
        "run_id": f"{CALCULUS}+3T2026",
        "start_date": "2020-01-01T00:00:00Z",
        "end_date": "2099-01-01T00:00:00Z",
        "status": "in progress",
        "dates_url": f"{COURSES_URL}/{CALCULUS}+3T2026/dates",
    }
    records["course_dates"].assert_not_called()


@pytest.mark.parametrize("found", ["xpro", None])
async def test_page_course_without_platform(mocker, records, found):
    """
    Without a platform, the course's own is looked up, and records are only
    read for a MITx Online course: a shared id could be another platform's.
    """
    lookup = mocker.patch("ai_chatbots.tools._get_course_platform", return_value=found)

    result = await call(get_course_progress, page_course=CALCULUS)

    lookup.assert_called_once_with(CALCULUS)
    assert result["enrollments"] == []
    assert "note" in result


async def test_page_course_with_platform_skips_lookup(mocker, records):
    """A platform sent with the request is used as is."""
    lookup = mocker.patch("ai_chatbots.tools._get_course_platform")

    result = await call(
        get_course_progress, page_course=CALCULUS, platform=MITXONLINE_PLATFORM
    )

    lookup.assert_not_called()
    assert result["enrollments"][0]["course_id"] == CALCULUS


async def test_page_course_not_enrolled_skips_lookup(mocker, records):
    """No platform is looked up for a course the learner isn't enrolled in."""
    lookup = mocker.patch("ai_chatbots.tools._get_course_platform")

    result = await call(get_course_progress, page_course="course-v1:MITx+X")

    lookup.assert_not_called()
    assert result["enrollments"] == []


@pytest.mark.parametrize("page_course", [CALCULUS, f"{CALCULUS}+1T2020"])
async def test_page_run_platform_is_its_course(mocker, settings, page_course):
    """
    A run id on the page looks up its course's platform, since the catalog only
    has courses. The real lookup runs; only its HTTP request is mocked.
    """
    settings.AI_MIT_LEARNING_RESOURCES_URL = "https://learn.test/resources/"
    cache = caches["default"]
    cache.clear()
    mocker.patch("ai_chatbots.tools.get_django_cache", return_value=cache)
    mocker.patch(
        "ai_chatbots.tools.learner_api.get_enrollments", return_value=ENROLLMENTS
    )
    response = mocker.Mock()
    response.json.return_value = {"results": [{"platform": {"code": "mitxonline"}}]}
    request = mocker.patch("ai_chatbots.learn_api.async_request", return_value=response)

    result = await call(get_course_progress, page_course=page_course)

    assert request.call_args.args[1]["readable_id"] == CALCULUS
    assert result["enrollments"][0]["course_id"] == CALCULUS


async def test_upcoming_assignments_skip_other_courses(records):
    """No due dates are read when the learner isn't enrolled in the page course."""
    result = await call(get_upcoming_assignments, page_course="course-v1:MITx+X")

    assert result["enrollments"] == []
    records["course_dates"].assert_not_called()


async def test_certificate_status_drops_personal_data(settings, records):
    """Program certificates keep their title, date and link, nothing personal."""
    settings.LEARNER_API_MITXONLINE_BASE_URL = "https://mitxonline.test/"
    result = await call(get_certificate_status, page_course=ECONOMICS)

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
        (get_enrollments, "programs"),
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


@pytest.mark.parametrize(
    ("certificate", "expected"),
    [
        (ISSUED_CERTIFICATE, "issued"),
        ({**ISSUED_CERTIFICATE, "verifiable_credential_json": None}, "not_confirmed"),
        ({**ISSUED_CERTIFICATE, "is_revoked": True}, "not_confirmed"),
        (LearnerAPIError("get_course_certificate", 404), "not_confirmed"),
    ],
)
async def test_digital_credential_issued_only_when_present(
    records, certificate, expected
):
    """A credential counts as issued only when MITx Online returns it."""
    if isinstance(certificate, Exception):
        records["course_certificate"].side_effect = certificate
    else:
        records["course_certificate"].return_value = certificate

    result = await call(get_certificate_status, page_course=ECONOMICS)

    (course,) = result["enrollments"]
    assert course["certificate_earned"] is True
    assert course["digital_credential"] == expected
    assert ("digital_credential_meaning" in result) is (expected == "not_confirmed")
    records["course_certificate"].assert_called_once_with("abc")
    assert "Jane Learner" not in json.dumps(result)


async def test_no_certificate_means_no_credential_lookup(records):
    """
    Without a certificate in the page's course there is nothing to look up or
    report, even when another course (14.750x) has one.
    """
    result = await call(get_certificate_status, page_course=CALCULUS)

    assert result["enrollments"][0]["certificate_earned"] is False
    assert "digital_credential" not in result["enrollments"][0]
    assert "digital_credential" not in result["programs"][0]
    assert "digital_credential_meaning" not in result
    records["course_certificate"].assert_not_called()
    records["program_certificate"].assert_not_called()


async def test_program_digital_credential(records):
    """Program certificates report their credential, without the uuid."""
    records["programs"].return_value = [
        {
            "program": {"title": "Data, Economics", "readable_id": PROGRAM},
            "certificate": {"link": "/certificate/program/def/", "uuid": "def"},
        }
    ]
    records["program_certificate"].return_value = {
        **ISSUED_CERTIFICATE,
        "verifiable_credential_json": None,
    }

    result = await call(get_certificate_status, page_course=ECONOMICS)

    (program,) = result["programs"]
    assert program["certificate_earned"] is True
    assert program["digital_credential"] == "not_confirmed"
    assert "couldn't be confirmed" in result["digital_credential_meaning"]
    assert "certificate_uuid" not in program
    assert "certificate_uuid" not in result["enrollments"][0]
    records["program_certificate"].assert_called_once_with("def")


@pytest.mark.parametrize("tool", [get_course_progress, get_upcoming_assignments])
@pytest.mark.parametrize("course_id", [None, "DEDP"])
async def test_program_page_isnt_a_course(records, tool, course_id):
    """A program's page, or the program by its short name, reads no course."""
    result = await call(
        tool, page_course=PROGRAM, platform=MITXONLINE_PLATFORM, course_id=course_id
    )

    assert result["enrollments"] == []
    assert result["note"].startswith("The page is a program")
    records["enrollments"].assert_not_called()


async def test_course_named_on_a_program_page(records):
    """A course the learner names on a program's page is read as usual."""
    result = await call(
        get_course_progress,
        page_course=PROGRAM,
        platform=MITXONLINE_PLATFORM,
        course_id="18.01x",
    )

    assert [e["course_id"] for e in result["enrollments"]] == [CALCULUS]


@pytest.mark.parametrize(
    ("page_course", "expected"),
    [(PROGRAM, PROGRAM), ("program-v1:MITx+SCM", "no MITx Online enrollment")],
)
async def test_certificate_status_names_the_page_program(
    records, page_course, expected
):
    """The page's program is pointed out, or said to be missing, never uncertified."""
    records["programs"].return_value[0]["certificate"] = {
        "link": "/certificate/program/def/",
        "uuid": "def",
    }

    result = await call(
        get_certificate_status, page_course=page_course, platform=MITXONLINE_PLATFORM
    )

    assert result["programs"][0]["certificate_earned"] is True
    assert expected in result["program_on_page"]


@pytest.mark.parametrize(
    ("platform", "found", "on_page"),
    [
        ("xpro", None, False),
        (None, "xpro", False),
        (None, None, False),
        (None, MITXONLINE_PLATFORM, True),
    ],
)
async def test_program_page_keeps_its_platform_check(
    mocker, records, platform, found, on_page
):
    """
    A program id doesn't say the platform, since xPRO's look like MITx Online's,
    so a program is only the page's once the page is known to be MITx Online.
    """
    lookup = mocker.patch("ai_chatbots.tools._get_course_platform", return_value=found)

    result = await call(get_certificate_status, page_course=PROGRAM, platform=platform)

    assert ("program_on_page" in result) is on_page
    assert result["note"].startswith("The page is a program") is on_page
    assert lookup.called is (platform is None)
    # Another platform's program reads none of the learner's MITx Online ones
    assert ("programs" in result) is on_page
    assert records["programs"].called is on_page
    assert records["program_certificates"].called is on_page


@pytest.mark.parametrize("page_course", [None, CALCULUS])
@pytest.mark.parametrize("course_id", [None, "18.01x"])
async def test_certificate_status_all_courses_ignores_the_page(
    records, page_course, course_id
):
    """all_courses lists every course's certificate, on any page."""
    result = await call(
        get_certificate_status,
        page_course=page_course,
        platform=MITXONLINE_PLATFORM,
        course_id=course_id,
        all_courses=True,
    )

    assert {e["course_id"] for e in result["enrollments"]} == {
        CALCULUS,
        ECONOMICS,
        ECONOMICS_2,
    }
    assert "note" not in result
    assert "course_on_page" not in result
    assert result["program_certificates"]


async def test_certificate_status_all_courses_on_another_platforms_page(records):
    """On an xPRO page, all_courses says the list can't answer for the page."""
    result = await call(
        get_certificate_status, page_course=PROGRAM, platform="xpro", all_courses=True
    )

    assert result["enrollments"]
    assert "isn't on MITx Online" in result["note"]


@pytest.mark.parametrize(
    ("page_course", "platform", "found"),
    [
        (SHARED, "xpro", MITXONLINE_PLATFORM),
        (CALCULUS, None, None),
    ],
)
async def test_certificate_status_on_another_platform_reads_no_programs(
    mocker, records, page_course, platform, found
):
    """A course that can't be checked here reads no programs or credentials."""
    mocker.patch("ai_chatbots.tools._get_course_platform", return_value=found)

    result = await call(
        get_certificate_status, page_course=page_course, platform=platform
    )

    assert set(result) == {"enrollments", "note"}
    assert "can't be checked here" in result["note"]
    for api in ("programs", "program_certificates", "program_certificate"):
        records[api].assert_not_called()
    records["course_certificate"].assert_not_called()


# A run whose id doesn't start with its course's (UAI.13), and the course its
# id does start with (UAI.9), as in MITx Online's records
UAI_9 = "course-v1:UAI_SOURCE+UAI.9"
UAI_13 = "course-v1:UAI_SOURCE+UAI.13"
UAI_RUN = f"{UAI_9}+3T2025"


def uai_records(*, in_page_run):
    """Enrollments in UAI.9 and UAI.13, in the page's UAI.13 run or another."""
    page_run = enrollment(UAI_13, "3T2025", "Generative AI")
    page_run["run"]["courseware_id"] = UAI_RUN
    return [
        enrollment(UAI_9, "1T2026", "Responsible AI"),
        page_run if in_page_run else enrollment(UAI_13, "1T2026", "Generative AI"),
    ]


@pytest.mark.parametrize(
    ("in_page_run", "course_id", "run_id"),
    [
        # The run's id names another course, which is the one asked about
        (True, "UAI.9", f"{UAI_9}+1T2026"),
        # The run's own course, known from the learner's enrollment in it
        (True, "UAI.13", UAI_RUN),
        # A run the learner isn't in has no known course; the name reads theirs
        (False, "UAI.13", f"{UAI_13}+1T2026"),
        (False, "UAI.9", f"{UAI_9}+1T2026"),
    ],
)
@pytest.mark.parametrize("tool", [get_course_progress, get_enrollments])
async def test_run_page_course_is_its_enrollments(
    records, in_page_run, course_id, run_id, tool
):
    """A run page's course comes from the learner's enrollment, not its id."""
    records["enrollments"].return_value = uai_records(in_page_run=in_page_run)

    result = await call(
        tool,
        page_course=UAI_RUN,
        platform=MITXONLINE_PLATFORM,
        course_id=course_id,
    )

    assert [e["run_id"] for e in result["enrollments"]] == [run_id]


@pytest.mark.parametrize(
    ("in_page_run", "expected"),
    [(True, "is enrolled in the course on this page"), (False, "no MITx Online")],
)
async def test_enrollments_say_whether_the_page_run_is_enrolled(
    records, in_page_run, expected
):
    """Asking whether they're enrolled in the page's run gets a yes or a no."""
    records["enrollments"].return_value = uai_records(in_page_run=in_page_run)

    result = await call(
        get_enrollments, page_course=UAI_RUN, platform=MITXONLINE_PLATFORM
    )

    assert len(result["enrollments"]) == 2
    assert expected in result["course_on_page"]


@pytest.mark.parametrize(
    ("page_course", "platform", "expected"),
    [
        (PROGRAM, MITXONLINE_PLATFORM, "is enrolled in the program on this page"),
        ("program-v1:MITx+SCM", MITXONLINE_PLATFORM, "no MITx Online enrollment"),
        (PROGRAM, "xpro", "isn't on MITx Online"),
    ],
)
async def test_enrollments_say_whether_the_page_program_is_enrolled(
    records, page_course, platform, expected
):
    """A program page says whether the learner is in it, not in a course."""
    result = await call(get_enrollments, page_course=page_course, platform=platform)

    assert expected in result["program_on_page"]
    assert "course_on_page" not in result


@pytest.mark.parametrize("course_id", ["DEDP", "Data, Economics", PROGRAM])
async def test_enrollments_for_a_named_program(records, course_id):
    """A program named by its number, title or id is listed, not said missing."""
    result = await call(get_enrollments, page_course=CALCULUS, course_id=course_id)

    assert result["enrollments"] == []
    assert [p["program_id"] for p in result["programs"]] == [PROGRAM]
    assert "note" not in result


@pytest.mark.parametrize("course_id", ["DEDP", PROGRAM])
async def test_enrollments_dont_name_another_platforms_page_program(records, course_id):
    """An xPRO program page's id names it, not a MITx Online program sharing it."""
    result = await call(
        get_enrollments, page_course=PROGRAM, platform="xpro", course_id=course_id
    )

    assert result == {"enrollments": [], "note": result["note"]}
    assert "isn't on MITx Online" in result["note"]


@pytest.mark.parametrize("tool", [get_enrollments, get_certificate_status])
async def test_page_program_by_title_keeps_its_platform_check(records, tool):
    """
    An xPRO program page's title names it, not a MITx Online program sharing
    its id.
    """
    result = await call(
        tool, page_course=PROGRAM, platform="xpro", course_id="Data, Economics"
    )

    assert result == {"enrollments": [], "note": result["note"]}
    assert "isn't on MITx Online" in result["note"]


@pytest.mark.parametrize("page_course", [PROGRAM, CALCULUS])
async def test_certificate_status_for_a_program_named_by_its_title(
    records, page_course
):
    """A program's title isn't said to be missing from the learner's records."""
    result = await call(
        get_certificate_status,
        page_course=page_course,
        platform=MITXONLINE_PLATFORM,
        course_id="Data, Economics",
    )

    assert result["programs"][0]["program_id"] == PROGRAM
    assert "is the learner's program Data, Economics" in result["note"]
    assert "isn't in the learner's MITx Online records" not in result["note"]


async def test_certificate_status_for_a_certificate_only_program(records):
    """A program certificate held only by MIT Learn isn't said to be missing."""
    records["programs"].return_value = []

    result = await call(
        get_certificate_status, page_course=CALCULUS, course_id="Data, Economics"
    )

    assert result["programs"] == []
    assert result["program_certificates"][0]["program_title"] == "Data, Economics"
    assert "is the learner's program Data, Economics" in result["note"]
    assert "isn't in the learner's MITx Online records" not in result["note"]
