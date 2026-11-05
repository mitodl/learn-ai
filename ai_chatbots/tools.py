"""Tools and schemas for AI agents"""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from urllib.parse import urljoin

import pydantic
from bs4 import BeautifulSoup
from django.conf import settings
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from ai_chatbots import learn_api, learner_api
from ai_chatbots.constants import (
    UAI_READABLE_ID_REGEX,
    ZENDESK_ARTICLE_SEARCH_PATH,
    ZENDESK_PLATFORM_CATEGORY_IDS,
    ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID,
    LearningResourceType,
    OfferedBy,
)
from ai_chatbots.utils import async_request, enum_zip, get_django_cache

log = logging.getLogger(__name__)

# Cache key prefix for the platform a course is offered on
COURSE_PLATFORM_CACHE_PREFIX = "course_platform_"

# Learn's platform code for MITx Online, the only platform the tools can read
MITXONLINE_PLATFORM = "mitxonline"

# What the LLM sees when a Learn-backed tool fails
LEARN_ERROR_ENVELOPE = json.dumps(
    {"error": "An error occurred while querying MIT Learn"}
)


def _search_error_envelope(error: Exception, **extra) -> str:
    """
    Serialize a failed search.  It keeps the url and parameters when the Learn
    client raised, so a failed search is distinguishable from one that ran and
    found nothing.
    """
    return json.dumps(
        {
            "error": "An error occurred while searching",
            "results": [],
            **extra,
            "metadata": {
                "search_url": getattr(error, "url", None),
                "parameters": getattr(error, "params", None),
            },
        }
    )


def _envelope(result: learn_api.SearchResult, shaped: list[dict], **extra) -> str:
    """Serialize shaped search results plus what was searched, for the LLM."""
    return json.dumps(
        {
            "results": shaped,
            "metadata": {"search_url": result.url, "parameters": result.params},
            **extra,
        }
    )


class SearchToolSchema(pydantic.BaseModel):
    """Schema to search for MIT learning resources.

    Attributes:
        q: The search query string
        resource_type: Filter by type of resource (course, program, etc)
        free: Filter for free resources only
        certification: Filter for resources offering certificates
        offered_by: Filter by institution offering the resource

    Here are some recommended tool parameters to apply for sample user prompts:

    User: "I am interested in learning advanced AI techniques for free"
    Search parameters: q="AI techniques", free=true

    User: "I am curious about AI applications for business"
    Search parameters: q="AI business"

    User: "I want free basic courses about biology from OpenCourseware"
    Search parameters: q="biology", resource_type=["course"], offered_by: ["ocw"]

    User: "I want to learn some advanced mathematics from MITx or OpenCourseware"
    Search parameters: q="mathematics", , offered_by: ["ocw", "mitx]

    """

    q: str = Field(
        description=(
            """The area of interest requested by the user.  NEVER INCLUDE WORDS SUCH AS
            "advanced" or "introductory" IN THIS PARAMETER! If the user asks for
            introductory, intermediate, or advanced courses, do not include that in the
            search query, but examine the search results to determine which most closely
            match the user's desired education level and/or their educational background
            (if either is provided) and choose those results to return to the user.  If
            the user asks what other courses are taught by a particular instructor,
            search the catalog for courses taught by that  instructor using the
            instructor's name as the value for this parameter.
            """
        )
    )

    state: Annotated[dict, InjectedState] = Field(
        description="The agent state, including any pinned search filters"
    )

    resource_type: list[enum_zip("resource_type", LearningResourceType)] | None = Field(
        default=None,
        description=(
            """
                Type of resource to search for: course, program, video, etc.
                If the user mentions courses, programs, videos, documents, or podcasts
                in particular, filter the search by this parameter.  DO NOT USE THE
                resource_type FILTER OTHERWISE. You MUST combine multiple resource types
                in one request like this: "resource_type=course&resource_type=program".
                Do not attempt more than one query per user message.
                If the user asks for podcasts, filter by both "podcast"
                and "podcast_episode".
                """
        ),
    )
    free: bool | None = Field(
        default=None,
        description=(
            """
            Whether the resource is free to access, true|false.
            true if the user is interested in free resources, false if the user is only
            interested in paid resources. Do not used this filter if the user does not
            indicate a preference.
            """
        ),
    )
    certification: bool | None = Field(
        default=None,
        description=(
            """
            Whether the resource offers a certificate upon completion, true|false.
            true if the user is interested in resources that offer certificates,
            false if the user does not want resources with a certificate offered.
            Do not use this filter if the user does not indicate a preference.
            """
        ),
    )
    offered_by: list[enum_zip("offered_by", OfferedBy)] | None = Field(
        default=None,
        description="""
            If a user asks for resources "offered by" or "from" an institution,
            you should include this parameter based on the following
            dictionary:

                mitx = "MITx"
                ocw = "MIT OpenCourseWare"
                bootcamps = "Bootcamps"
                xpro = "MIT xPRO"
                mitpe = "MIT Professional Education"
                see = "MIT Sloan Executive Education"

            DON'T USE THE offered_by FILTER OTHERWISE.
            Combine 2+ offered_by values in 1 query.
            """,
    )


@tool(args_schema=SearchToolSchema)
async def search_courses(
    q: str, state: Annotated[dict, InjectedState] | None, **kwargs
) -> str:
    """
    Query the MIT API for learning resources, and
    return simplified results as a JSON string
    """
    filters = {
        "resource_type": [rt.name for rt in (kwargs.get("resource_type") or [])]
        or None,
        "free": kwargs.get("free"),
        "offered_by": [o.name for o in (kwargs.get("offered_by") or [])] or None,
        "certification": kwargs.get("certification"),
    }
    filters = {k: v for k, v in filters.items() if v is not None}
    # Filters pinned by the client (e.g. a channel page) replace any the agent
    # chose for the same field, so results can't fall outside them.
    search_filters = (state or {}).get("search_filters") or []
    if search_filters:
        filters.update(search_filters[-1])
    try:
        result = await learn_api.search_learning_resources(q, filters)
        main_properties = [
            "title",
            "id",
            "readable_id",
            "description",
            "offered_by",
            "free",
            "certification",
            "resource_type",
        ]
        shaped = []
        for row in result.results:
            simplified = {k: row.get(k) for k in main_properties}
            simplified["url"] = f"{settings.AI_MIT_SEARCH_DETAIL_URL}{row.get('id')}"
            # Instructors and level will be in the runs data if present
            next_date = row.get("next_start_date")
            raw_runs = row.get("runs", [])
            best_run = None
            if next_date:
                runs = [run for run in raw_runs if run.get("start_date") == next_date]
                if runs:
                    best_run = runs[0]
            elif raw_runs:
                best_run = raw_runs[-1]
            if best_run:
                for attribute in ("level", "instructors"):
                    simplified[attribute] = best_run.get(attribute, [])
            shaped.append(simplified)
        return _envelope(result, shaped)
    except Exception as error:
        log.exception("Error querying MIT API")
        return _search_error_envelope(error)


class SearchContentFilesToolSchema(pydantic.BaseModel):
    """
    Schema for searching MIT contentfiles related to a particular learning
    resource, including its price, dates, format, instructors and
    prerequisites.
    """

    q: str = Field(
        description=("Query to find requested information about a learning resource.")
    )

    readable_id: str | None = Field(
        description=("The readable_id of the learning resource."),
        default=None,
    )

    state: Annotated[dict, InjectedState] = Field(
        "Agent state, which may include course_id (readable_id) and collection_name"
    )


class SearchRelatedCourseContentFilesToolSchema(pydantic.BaseModel):
    """
    Search for information on courses within a program.
    This can be used if the user asks for
    information specific to one course or many courses in a program
    """

    q: str = Field(
        description=(
            "Query content related to course(s) in a program that "
            "might answer the user's question."
        )
    )
    state: Annotated[dict, InjectedState] = Field(
        description=(
            "The agent state with a related_courses param to query "
            "content related to course(s) in a program"
        )
    )


class VideoGPTToolSchema(pydantic.BaseModel):
    """Schema for searching MIT contentfiles for to a particular video transcript."""

    q: str = Field(
        description=(
            "Query to find transcript information that might answer the user's\
                question."
        )
    )
    state: Annotated[dict, InjectedState] = Field(
        description="The agent state, including video transcript block id"
    )


WITHHELD_LINKS_NOTE = (
    "Results with a null id have no url. Answer from them as usual, but never "
    "write about links, urls or downloads for them: no links, no placeholder "
    "links, and no remark that a link is unavailable, missing or not provided. "
    "If the user asks for a link to one of them, answer without commenting on "
    "links."
)


async def _enrolled_runs() -> set[str]:
    """
    Get the MITx Online runs the signed-in learner is enrolled in, read with
    their own session. Empty with no session or if the records can't be read.
    """
    if not learner_api.has_learner_session():
        return set()
    runs = set()
    try:
        for enrollment in await learner_api.get_enrollments():
            run = enrollment.get("run") if isinstance(enrollment, dict) else None
            run_id = run.get("courseware_id") if isinstance(run, dict) else None
            if run_id and isinstance(run_id, str):
                runs.add(run_id)
    except Exception:  # noqa: BLE001
        log.warning("Couldn't read the learner's enrollments to check course links")
        return set()
    return runs


async def _content_file_search(
    q: str,
    readable_ids: str | list[str],
    *,
    collection_name: str | None,
    exclude_canvas: str = "True",
    platform: str | None = None,
) -> str:
    """Search content files and shape chunks + citations for the LLM."""
    try:
        result = await learn_api.search_content_files(
            q, readable_ids, collection_name=collection_name, platform=platform
        )
        # Consumers store the flag as a string, evaluation sets as a bool
        exclude = str(exclude_canvas) != "False"
        shaped = []
        citations = {}
        enrolled_runs = None
        links_withheld = False
        for chunk in result.results:
            platform = chunk.get("platform", {}).get("code")
            # Currently, canvas contentfiles have blank platform values,
            # those from other sources do not.
            if exclude and (not platform or platform == "canvas"):
                continue
            # the citation url is what identifies a citation; resource_point_id
            # is shared by every chunk of every file in a course, so keying on
            # it collapsed every citation onto whichever result came back first
            citation_key = chunk.get("url")
            # Link a file from an Open edX course (it has an edx_module_id;
            # Canvas files do too, but stay in Canvas) only for learners
            # enrolled in its run, since most of them open only for enrolled
            # learners. Only MITx Online enrollments can be checked.
            is_course_material = (
                chunk.get("edx_module_id") and platform and platform != "canvas"
            )
            if citation_key and is_course_material:
                if platform == MITXONLINE_PLATFORM and enrolled_runs is None:
                    enrolled_runs = await _enrolled_runs()
                if (
                    platform != MITXONLINE_PLATFORM
                    or chunk.get("run_readable_id") not in enrolled_runs
                ):
                    citation_key = None
                    links_withheld = True
            shaped.append(
                {
                    "id": citation_key,
                    "chunk_content": chunk.get("chunk_content"),
                    "run_title": chunk.get("run_title"),
                }
            )
            if citation_key and not citations.get(citation_key):
                citations[citation_key] = {
                    "citation_url": chunk.get("url"),
                    "citation_title": chunk.get("title") or chunk.get("content_title"),
                }
        # Otherwise the model offers the files anyway, with links to "#"
        note = {"note": WITHHELD_LINKS_NOTE} if links_withheld else {}
        return _envelope(result, shaped, citation_sources=citations, **note)
    except Exception as error:
        log.exception("Error querying MIT API")
        return _search_error_envelope(error, citation_sources={})


@tool(args_schema=SearchContentFilesToolSchema)
async def search_content_files(
    q: str, state: Annotated[dict, InjectedState], readable_id: str | None = None
) -> str:
    """
    Search the content of one particular MIT learning resource, identified by
    its readable_id or course_id.  It covers what the resource teaches, its
    schedule, format, duration, instructors and prerequisites, what it costs,
    whether it has a free or audit option, what a certificate for it costs, and
    its enrollment and payment deadlines.  Use this first for any question that
    could be specific to this resource.
    """
    return await _content_file_search(
        q,
        state.get("course_id", [None])[-1] or readable_id,
        collection_name=state.get("collection_name", [None])[-1],
        exclude_canvas=state.get("exclude_canvas", ["True"])[-1],
        # A readable id can match resources on two platforms; scope the search
        # to the one the request named so the other one's content can't answer.
        platform=(state.get("platform") or [None])[-1],
    )


@tool(args_schema=SearchRelatedCourseContentFilesToolSchema)
async def search_related_course_content_files(
    q: str, state: Annotated[dict, InjectedState]
) -> str:
    """
    Search for information across related courses in the same program.
    Use this tool when the user's question may be answered by content from
    other courses in the program, or when the current course search does not
    return sufficient results.
    """
    return await _content_file_search(
        q,
        state["related_courses"],
        collection_name=state.get("collection_name", [None])[-1],
        exclude_canvas=state.get("exclude_canvas", ["True"])[-1],
    )


@tool(args_schema=VideoGPTToolSchema)
async def get_video_transcript_chunk(
    q: str, state: Annotated[dict, InjectedState]
) -> str:
    """
    Query the MIT video transcript API, and return results as a JSON string.
    """
    try:
        result = await learn_api.search_transcript(q, state["transcript_asset_id"][-1])
        shaped = [{"chunk_content": r.get("chunk_content")} for r in result.results]
        return _envelope(result, shaped)
    except Exception:
        log.exception("Error querying MIT API for transcripts")
        return LEARN_ERROR_ENVELOPE


class SearchSupportArticlesToolSchema(pydantic.BaseModel):
    """
    Schema to search the MIT Learn support center (Zendesk help center) for
    articles about using MIT platforms and accounts: transactions, access
    problems, technical problems and platform-wide policies.  It does not cover
    what a particular resource costs, requires or offers, or when it runs.

    Here are some recommended tool parameters to apply for sample user prompts.
    Keep the distinctive words of the problem, since this is a keyword search:

    User: "How can I pay for this course?"
    Search parameters: q="payment"

    User: "I cannot get to the certificate I earned"
    Search parameters: q="certificate"

    User: "I paid for the course but am not enrolled in it"
    Search parameters: q="paid not enrolled"

    User: "The payment page will not load when I try to upgrade"
    Search parameters: q="payment page not loading upgrade"

    User: "I cannot log in to MIT Learn"
    Search parameters: q="login"

    User: "How do I unenroll?"
    Search parameters: q="unenroll"

    User: "Can I get credit for OpenCourseWare courses?"
    Search parameters: q="credit"

    User: "I can't log in to MIT Learn"
    Search parameters: q="login"

    User: "How do refunds work?"
    Search parameters: q="refund"
    """

    q: str = Field(
        description=(
            """
            Keywords describing the support question, i.e. "certificate", "refund",
            "unenroll", "password reset".  Use a few keywords rather than a full
            sentence, since this is a keyword search rather than a semantic one.
            Keep the user's own distinctive words - "paid not enrolled", "payment
            page not loading" - rather than replacing them with a category, since
            "enrollment issue" or "payment problem" match almost any article.
            """
        )
    )

    state: Annotated[dict, InjectedState] = Field(
        description=(
            "The agent state, which may include the course_id of the resource "
            "under discussion. Its platform and readable id determines "
            "which part of the support center is searched."
        )
    )


def _simplify_zendesk_article(article: dict) -> dict:
    """
    Convert a Zendesk help center article into a simplified dict,
    with the html body converted to truncated plain text.
    """
    body_text = BeautifulSoup(article.get("body") or "", "html.parser").get_text(
        " ", strip=True
    )
    return {
        "id": str(article.get("id")),
        "title": article.get("title"),
        "url": article.get("html_url"),
        "updated_at": article.get("updated_at"),
        "content": body_text[: settings.AI_ZENDESK_ARTICLE_MAX_CHARS],
    }


async def _get_course_platform(readable_id: str) -> str | None:
    """
    Return the MIT Learn platform code of a course, or None if it cannot be
    determined.  Cached, since a course does not change platforms.
    """
    cache = get_django_cache()
    cache_key = f"{COURSE_PLATFORM_CACHE_PREFIX}{readable_id}"
    cached_platform = await cache.aget(cache_key)
    if cached_platform is not None:
        # Failed and empty lookups are cached as an empty string
        return cached_platform or None
    try:
        results = await learn_api.learning_resources(readable_id)
        codes = {(r.get("platform") or {}).get("code") for r in results} - {None}
        # A readable id can match resources on two platforms.  Picking one of
        # them scopes the support search to the wrong help center section, so
        # stay unscoped unless the matches agree.
        platform = codes.pop() if len(codes) == 1 else None
    except Exception:
        log.exception("Error looking up the platform of course %s", readable_id)
        # Cache the failure briefly, so that an unreachable API is not retried
        # in front of every support question while it is down.
        await cache.aset(
            cache_key, "", settings.AI_COURSE_PLATFORM_ERROR_CACHE_DURATION
        )
        return None
    await cache.aset(
        cache_key, platform or "", settings.AI_COURSE_PLATFORM_CACHE_DURATION
    )
    return platform


def _zendesk_categories(course_id: str | None, platform: str | None) -> list[str]:
    """
    Return the zendesk help center categories that cover a course, so that a
    search can be limited to the parts of the support center relevant to it.
    """
    categories = []
    platform_category = ZENDESK_PLATFORM_CATEGORY_IDS.get(platform)
    if platform_category:
        categories.append(platform_category)
    if course_id and UAI_READABLE_ID_REGEX.match(course_id):
        # UAI courses are on MITx Online but documented under their own
        # category, so both are searched.
        categories.append(ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID)
    return categories


def _support_output(
    portal_url: str | None,
    articles: list[dict] | None = None,
    metadata: dict | None = None,
) -> str:
    """
    Serialize the support search result.  The support center url is keyed by
    what to do with it, because the model ignores the same instruction when it
    is prose in the prompt or the tool docstring.
    """
    articles = articles or []
    return json.dumps(
        {
            "results": articles,
            "support_center_url_to_offer_if_no_article_answers": portal_url,
            "citation_sources": {
                article["id"]: {
                    "citation_url": article["url"],
                    "citation_title": article["title"],
                }
                for article in articles
                if article["url"]
            },
            "metadata": metadata or {},
        }
    )


@tool(args_schema=SearchSupportArticlesToolSchema)
async def search_support_articles(q: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Search the MIT Learn support center (help center) for up to date articles
    about using MIT platforms and accounts: carrying out a transaction such as
    paying, enrolling, unenrolling or requesting a refund, reaching something
    already earned or paid for, signing in and other technical problems, and
    platform-wide policies such as accessibility and how long access to course
    material lasts.  It answers how to do something or what to do when it does
    not work; what a resource costs, covers, requires or offers, and when it
    runs, are in the resource's own content.  Returns the articles as a JSON
    string, with a support center url to offer the user when none of the
    articles it found answers their question.
    """
    portal_url = settings.AI_ZENDESK_URL
    if not portal_url:
        log.warning("No support portal url is configured")
        return _support_output(None)

    search_url = f"{portal_url.rstrip('/')}{ZENDESK_ARTICLE_SEARCH_PATH}"
    params = {"query": q, "per_page": settings.AI_ZENDESK_SEARCH_LIMIT}

    # Limit the search to the parts of the support center covering the course
    # under discussion.
    course_ids = (state or {}).get("course_id") or [None]
    course_id = course_ids[-1]
    # A readable id can match resources on two platforms, so trust the
    # platform the request supplied over looking one up by readable id.
    platform = ((state or {}).get("platform") or [None])[-1]
    if not platform and course_id:
        platform = await _get_course_platform(course_id)
    categories = _zendesk_categories(course_id, platform)
    if categories:
        # The search endpoint takes a comma separated list of category ids
        params["category"] = ",".join(categories)
    else:
        # Without a category the whole support center is searched, which risks
        # answering with an article about some other MIT platform.
        log.info(
            "Searching the whole support center; no category is mapped to the "
            "platform (%s) of course %s",
            platform,
            course_id,
        )

    try:
        # This is a public help center, so no authentication is sent
        response = await async_request(
            search_url, params, timeout=settings.REQUESTS_TIMEOUT
        )
        response.raise_for_status()
        articles = [
            _simplify_zendesk_article(article)
            for article in response.json().get("results", [])
        ]
    except Exception:
        log.exception("Error querying the support portal at %s", search_url)
        return _support_output(portal_url)

    if categories and not articles:
        # A stale category id is not an error: zendesk answers 200 with an
        # empty result set, so every scoped search would quietly return
        # nothing.  Log it, since only repeated misses distinguish a bad
        # category from a query with no matching article.
        log.warning(
            "No support articles found in categories %s for the platform (%s) "
            "of course %s",
            params["category"],
            platform,
            course_id,
        )

    return _support_output(
        portal_url,
        articles,
        metadata={
            "search_url": search_url,
            "parameters": params,
            "platform": platform,
        },
    )


# Account tools: read the signed-in learner's own records with learner_api.
# The learner's session comes from the request (see learner_api), never from
# tool arguments or state, and only the fields an answer needs reach the model.

COULD_NOT_CHECK = (
    "The learner's records could not be checked right now. Tell them you "
    "couldn't check, and never conclude they have no enrollment, grade or "
    "certificate."
)
DIGITAL_CREDENTIAL_ISSUED = "issued"
DIGITAL_CREDENTIAL_NOT_CONFIRMED = "not_confirmed"


class AccountToolSchema(pydantic.BaseModel):
    """Schema for tools that read the signed-in learner's own records."""

    state: Annotated[dict, InjectedState] = Field(
        description="The agent state, which includes the course_id of the page."
    )


class CourseAccountToolSchema(AccountToolSchema):
    """Schema for account tools that read one course's records."""

    user_course_input: str | None = Field(
        default=None,
        description=(
            """
            Only when the learner asks about a course other than the one on
            this page: the course as they named it, its number, id or title
            (such as "18.01x" or "Calculus 1A"), exactly as they wrote it.
            Don't look it up or guess a number for a title. Leave it out for
            the course on this page.
            """
        ),
    )


class CertificateToolSchema(CourseAccountToolSchema):
    """Schema for get_certificate_status, which can also list every certificate."""

    all_courses: bool = Field(
        default=False,
        description=(
            """
            True only when the learner asks about all of their certificates,
            such as "what certificates do I have". Never for a question about
            one course or program, including the one on this page ("did I get
            my certificate for this program"). The course on this page and
            user_course_input are then ignored.
            """
        ),
    )


class EnrollmentsToolSchema(AccountToolSchema):
    """Schema for get_enrollments, which lists all courses or a named one."""

    user_course_input: str | None = Field(
        default=None,
        description=(
            """
            The course the learner asks about, the one on this page included:
            as they named it, its number, id or title (such as "18.01x" or
            "Calculus 1A"), exactly as they wrote it. Leave it out only when
            they ask about all of their courses.
            """
        ),
    )


def _parse_date(value: str | None) -> datetime | None:
    """Parse an ISO date from the APIs as aware UTC, or None."""
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    # A date with no offset can't be compared with an aware "now"
    if parsed and parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _run_status(run: dict, now: datetime) -> str:
    """Say whether a course run is upcoming, in progress or ended."""
    start, end = _parse_date(run.get("start_date")), _parse_date(run.get("end_date"))
    if start and start > now:
        return "upcoming"
    if end and end < now:
        return "ended"
    return "in progress"


def _course_tab_url(run: dict, tab: str) -> str | None:
    """Link to a tab (progress, dates) of the run's course, if it can be built."""
    url = run.get("courseware_url")
    if url and re.search(r"/home/?$", url):
        return re.sub(r"/home/?$", f"/{tab}", url)
    return None


def _certificate_url(certificate) -> str | None:
    """Make a MITx Online certificate's link (a path) absolute."""
    link = certificate.get("link") if isinstance(certificate, dict) else None
    return urljoin(settings.LEARNER_API_MITXONLINE_BASE_URL, link) if link else None


def _certificate_uuid(certificate) -> str | None:
    """Return a MITx Online certificate's uuid, used to look up its credential."""
    return certificate.get("uuid") if isinstance(certificate, dict) else None


def _enrollment_summary(enrollment: dict, now: datetime) -> dict:
    """Keep the parts of a MITx Online enrollment that an answer needs."""
    run = enrollment.get("run") or {}
    course = run.get("course") or {}
    grades = enrollment.get("grades") or []
    grade = grades[0] if grades else None
    # grade_percent reads 0 when no grade is recorded, so it's only used
    # alongside a recorded grade (a 0-1 fraction).
    if grade and grade.get("grade") is None:
        grade = None
    certificate = enrollment.get("certificate")
    return {
        "course_title": course.get("title") or run.get("title"),
        "course_id": course.get("readable_id"),
        "run_id": run.get("courseware_id"),
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "status": _run_status(run, now),
        "self_paced": run.get("is_self_paced"),
        "enrollment_mode": enrollment.get("enrollment_mode"),
        "grade": {
            "percent": grade.get("grade_percent"),
            "letter": grade.get("letter_grade"),
            "passed": grade.get("passed"),
        }
        if grade
        else None,
        "certificate_earned": bool(certificate),
        "certificate_url": _certificate_url(certificate),
        "certificate_uuid": _certificate_uuid(certificate),
        "course_url": run.get("courseware_url"),
        "progress_url": _course_tab_url(run, "progress"),
        "dates_url": _course_tab_url(run, "dates"),
    }


def _current_runs(
    summaries: list[dict], *, prefer_certificate: bool = False
) -> list[dict]:
    """
    Keep one enrollment per course: the run in progress, else the latest. With
    prefer_certificate, a run that earned a certificate wins, so a newer run
    doesn't hide a certificate from an earlier one.
    """
    order = {"in progress": 2, "upcoming": 1, "ended": 0}

    def rank(summary):
        return (
            prefer_certificate and summary["certificate_earned"],
            order[summary["status"]],
            _parse_date(summary["start_date"]) or datetime.min.replace(tzinfo=UTC),
        )

    best: dict[str, dict] = {}
    for summary in summaries:
        key = summary["course_id"] or summary["run_id"]
        if key not in best or rank(summary) > rank(best[key]):
            best[key] = summary
    return list(best.values())


async def _summaries() -> list[dict]:
    """
    Get a summary of each of the learner's MITx Online enrollments.

    Raises:
        learner_api.LearnerAPIError: if the records can't be read
    """
    now = datetime.now(tz=UTC)
    return [
        _enrollment_summary(enrollment, now)
        for enrollment in await learner_api.get_enrollments()
    ]


class AccountLookupStatus(StrEnum):
    """Mutually exclusive outcomes of selecting course or program account records."""

    MATCHED = "matched"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    NOT_ENROLLED_ON_PAGE = "not_enrolled_on_page"
    NOT_ON_MITXONLINE = "not_on_mitxonline"
    ON_PROGRAM_PAGE = "on_program_page"
    PLATFORM_UNCLEAR = "platform_unclear"
    NO_PAGE_COURSE = "no_page_course"

    @property
    def note(self) -> str | None:
        """Default model guidance; named-course outcomes need a specific note."""
        return {
            self.NOT_ENROLLED_ON_PAGE: (
                "The learner has no MITx Online enrollment in the course on this "
                "page. Don't answer from their other courses."
            ),
            self.NOT_ON_MITXONLINE: (
                "The course on this page isn't on MITx Online, so the learner's "
                "records for it can't be checked here."
            ),
            self.ON_PROGRAM_PAGE: (
                "The page is a program, not a course. For its certificate, see "
                "programs and program_certificates; for one of its courses, pass "
                "that course as user_course_input."
            ),
            self.PLATFORM_UNCLEAR: (
                "It isn't clear which platform the course on this page is on, so "
                "the learner's records for it can't be checked here."
            ),
            self.NO_PAGE_COURSE: (
                "There is no course on this page. Pass the course the learner "
                "asked about as user_course_input."
            ),
        }.get(self)


def _empty_course_selection(
    status: AccountLookupStatus, note: str | None = None
) -> dict:
    """Keep selection state separate from the instructions given to the model."""
    return {
        "enrollments": [],
        "selection_status": status,
        "note": note if note is not None else status.note,
    }


# MITx Online's program ids, which xPRO's share, so it doesn't say the platform
PROGRAM_ID_PREFIX = "program-v1:"
# A run id. Most don't start with their course's readable id, so its course
# is only known from the learner's enrollment in it.
RUN_ID_REGEX = re.compile(r"course-v1:[^+]+\+[^+]+\+[^+]+")


async def _page_platform_status(
    platform: str | None, resource_id: str
) -> AccountLookupStatus | None:
    """Return a platform restriction on the page's course or program, or None."""
    if not platform:
        # A readable id can be shared by courses on two platforms, so without
        # one the MITx Online records could be another platform's course. The
        # catalog only knows courses, so look up the course, not the run.
        platform = await _get_course_platform(resource_id)
        if platform is None:
            return AccountLookupStatus.PLATFORM_UNCLEAR
    if platform != MITXONLINE_PLATFORM:
        return AccountLookupStatus.NOT_ON_MITXONLINE
    return None


def _get_page_context(
    state: dict, summaries: list[dict] = ()
) -> tuple[dict, str | None]:
    """
    Read the page's resource ID and platform from agent state. Keep course,
    run and program IDs separate; a run's course comes from its enrollment.
    """
    state = state or {}
    page_resource_id = (state.get("course_id") or [None])[-1]
    run_id = (
        page_resource_id if RUN_ID_REGEX.fullmatch(page_resource_id or "") else None
    )
    program_id = (
        page_resource_id
        if page_resource_id and page_resource_id.startswith(PROGRAM_ID_PREFIX)
        else None
    )
    course_id = None if run_id or program_id else page_resource_id
    if run_id:
        course_id = next(
            (s["course_id"] for s in summaries if s["run_id"] == run_id), None
        )
    page = {
        "resource_id": page_resource_id,
        "course_id": course_id,
        "run_id": run_id,
        "program_id": program_id,
    }
    return page, (state.get("platform") or [None])[-1]


def _page_courses(page: dict, summaries: list[dict]) -> set[str]:
    """Return the course ids that name the page's course, under any of its runs."""
    page_resource_id = page["resource_id"]
    if not page_resource_id:
        return set()
    return {page_resource_id, page["course_id"]} - {None} | {
        summary["course_id"]
        for summary in summaries
        if page_resource_id in (summary["course_id"], summary["run_id"])
    }


def _normalize_course_input(user_course_input: str | None) -> str | None:
    """Strip whitespace, quotes and sentence punctuation from both ends."""
    if user_course_input is None:
        return None
    # A leading "." stays, for names like ".NET"; "+" is never stripped.
    quotes = " \t\n\"'\u201c\u201d\u2018\u2019"
    query = user_course_input.lstrip(quotes + "?!,;").rstrip(quotes + "?!.,;")
    # An unusable explicit name must not turn into a request for the page course.
    return query or user_course_input


def _matches_resource_query(resource_query: str, summary: dict) -> bool:
    """
    Match a course or program reference against the record's:
    - run id, whole only, since a run tag like "1T2025" ends many of them;
    - course id, whole or its end after ":" or "+", with or without the
      number's trailing "x" ("18.01x" or "18.01", not "8.01x");
    - title, on whole words ("Calculus 1A", not "Calculus 1").
    """
    phrase = r"\s+".join(map(re.escape, resource_query.split()))
    if not phrase:
        return False
    query = resource_query.casefold()
    resource_id = (
        summary.get("course_id") or summary.get("program_id") or ""
    ).casefold()
    return (
        query in (resource_id, (summary.get("run_id") or "").casefold())
        or resource_id.endswith(
            tuple(f"{sep}{query}{x}" for sep in ":+" for x in ("", "x"))
        )
        or bool(
            phrase
            and re.search(
                rf"(?<!\w){phrase}(?!\w)",
                summary.get("course_title") or summary.get("program_title") or "",
                re.IGNORECASE,
            )
        )
    )


def _not_in_records(user_course_input: str) -> str:
    """Say a named resource isn't in the records, not that the learner lacks it."""
    return (
        f"{user_course_input} isn't in the learner's MITx Online records. Say you "
        "don't see it there and can't check other platforms; don't say they have "
        "no enrollment or certificate, and don't answer from their other courses "
        "or programs."
    )


# For tools that read one course, where get_enrollments can find the right one
LIST_COURSES_HINT = " get_enrollments lists their courses and their course_id."


def _named_programs(program_query: str, programs: list[dict]) -> list[dict]:
    """Match a free-text program reference against program IDs and titles."""
    return [p for p in programs if _matches_resource_query(program_query, p)]


async def _page_program_status(
    state: dict, programs: list[dict]
) -> AccountLookupStatus | None:
    """Check the platform when a named program is also the page's program."""
    page, platform = _get_page_context(state)
    if page["program_id"] and any(
        p["program_id"] == page["program_id"] for p in programs
    ):
        return await _page_platform_status(platform, page["program_id"])
    return None


def _names_only_the_page_course(
    course_query: str, matches: list[dict], page_courses: set[str]
) -> bool:
    """
    Say whether the course a learner named, such as by its title, is the page's
    course under another name. A run asked for by its id isn't: it keeps that run.
    """
    return all(s["course_id"] in page_courses for s in matches) and (
        course_query.casefold() not in {(s["run_id"] or "").casefold() for s in matches}
    )


async def _course_enrollments(  # noqa: C901, PLR0911, PLR0912
    state: dict,
    user_course_input: str | None = None,
    *,
    prefer_certificate: bool = False,
) -> dict:
    """
    Select the named course's records, or default to the page's course or run.
    Return a selection_status independently of any readable note. See
    _matches_resource_query for how a named course is matched.

    Raises:
        learner_api.LearnerAPIError: if the records can't be read
    """
    course_query = _normalize_course_input(user_course_input)
    use_page_context = not course_query
    page_summary, platform = _get_page_context(state)
    page_resource_id = page_summary["resource_id"]
    summaries = None
    if course_query and page_summary["run_id"] and not page_summary["course_id"]:
        # A run's course, which the name could be, comes from its enrollment
        summaries = await _summaries()
        page_summary, _ = _get_page_context(state, summaries)
    if (
        course_query
        and page_resource_id
        and _matches_resource_query(course_query, page_summary)
    ):
        # A short name for the page's course must keep its platform check.
        use_page_context = True
    if use_page_context and not page_resource_id:
        return _empty_course_selection(AccountLookupStatus.NO_PAGE_COURSE)
    if use_page_context and platform and platform != MITXONLINE_PLATFORM:
        return _empty_course_selection(AccountLookupStatus.NOT_ON_MITXONLINE)
    if use_page_context and page_resource_id.startswith(PROGRAM_ID_PREFIX):
        status = await _page_platform_status(platform, page_resource_id)
        return _empty_course_selection(status or AccountLookupStatus.ON_PROGRAM_PAGE)
    if summaries is None:
        summaries = await _summaries()
    matches = [
        summary
        for summary in summaries
        if (
            page_resource_id in (summary["course_id"], summary["run_id"])
            if use_page_context
            else _matches_resource_query(course_query, summary)
        )
    ]
    if not matches and not use_page_context:
        # Certificate lookup must also consider programs before reporting a miss.
        return {"enrollments": [], "selection_status": AccountLookupStatus.NOT_FOUND}
    if not matches:
        return _empty_course_selection(AccountLookupStatus.NOT_ENROLLED_ON_PAGE)
    named = {summary["course_id"] or summary["run_id"] for summary in matches}
    if not use_page_context and len(named) > 1:
        return _empty_course_selection(
            AccountLookupStatus.AMBIGUOUS,
            f"{user_course_input} names more than one of the learner's courses: "
            f"{', '.join(sorted(named))}. Ask which one they mean, then pass "
            "its course_id as user_course_input.",
        )
    page_courses = _page_courses(page_summary, summaries)
    # Another course is MITx Online's own record, but the page's course, under
    # any of its ids, could be a different platform's course with the same id.
    if use_page_context or any(s["course_id"] in page_courses for s in matches):
        status = await _page_platform_status(platform, matches[0]["course_id"])
        if status:
            return _empty_course_selection(status)
    if not use_page_context and _names_only_the_page_course(
        course_query, matches, page_courses
    ):
        # A title names the page's course; an explicit run ID keeps that run.
        use_page_context = True
        matches = [
            s for s in summaries if page_resource_id in (s["course_id"], s["run_id"])
        ]
        if not matches:
            return _empty_course_selection(AccountLookupStatus.NOT_ENROLLED_ON_PAGE)
    result = {
        "enrollments": _current_runs(matches, prefer_certificate=prefer_certificate),
        "selection_status": AccountLookupStatus.MATCHED,
    }
    if use_page_context:
        result["page_resource_id"] = page_resource_id
    return result


def _could_not_check(operation: str) -> str:
    """Report a failed lookup without suggesting the learner has no records."""
    log.warning("Account tool %s couldn't read the learner's records", operation)
    return json.dumps({"error": COULD_NOT_CHECK})


def _only(summaries: list[dict], *fields: str) -> list[dict]:
    """Keep the named fields, plus what identifies the course."""
    keep = ("course_title", "course_id", "run_id", *fields)
    return [{field: summary[field] for field in keep} for summary in summaries]


@tool(args_schema=EnrollmentsToolSchema)
async def get_enrollments(  # noqa: C901
    state: Annotated[dict, InjectedState], user_course_input: str | None = None
):
    """
    Get the courses the signed-in learner is enrolled in on MITx Online: each
    course run's dates, whether it is upcoming, in progress or ended, the
    enrollment mode (audit or verified) and the course's certificate, and the
    programs they are enrolled in, with whether they earned each program's
    certificate. A course's certificate can be from an earlier run than the
    one listed; its run_id says which. Use it for "what am I enrolled in",
    "when does my course start" or "am I enrolled in X". It returns all of the
    learner's courses and programs, not just the one on this page, or only the
    user_course_input given. For "what certificates do I have", use
    get_certificate_status with all_courses instead: only it lists program
    certificates from MIT Learn and digital credentials.
    """
    try:
        summaries, programs = await asyncio.gather(
            _summaries(), learner_api.get_program_enrollments()
        )
    except Exception:  # noqa: BLE001
        return _could_not_check("get_enrollments")
    programs = [
        {k: v for k, v in p.items() if k != "certificate_uuid"}
        for p in map(_program_enrollment_summary, programs)
    ]
    page, platform = _get_page_context(state, summaries)
    course_query = _normalize_course_input(user_course_input)
    page_resource_id = page["resource_id"]
    page_program = next(
        (p for p in programs if p["program_id"] == page_resource_id), None
    )
    if course_query:
        page_courses = _page_courses(page, summaries)
        # A name for the page's course is the page's course, not another
        # course with the same number
        names_page = bool(page_resource_id) and _matches_resource_query(
            course_query, page
        )
        summaries = [
            s
            for s in summaries
            if _matches_resource_query(course_query, s)
            and (not names_page or s["course_id"] in page_courses)
        ]
        # The page's course, under any of its names, could be another
        # platform's with the same id
        on_page = [s for s in summaries if s["course_id"] in page_courses]
        # A run the learner isn't in has no known course to look up
        page_id = (on_page[0] if on_page else page)["course_id"]
        if (
            (names_page or on_page)
            and page_id
            and (status := await _page_platform_status(platform, page_id))
        ):
            return json.dumps(_empty_course_selection(status))
        programs = _named_programs(course_query, programs)
        if status := await _page_program_status(state, programs):
            return json.dumps(_empty_course_selection(status))
    result = {"enrollments": _current_runs(summaries), "programs": programs}
    if course_query and not summaries and not programs:
        result["selection_status"] = AccountLookupStatus.NOT_FOUND
        result["note"] = _not_in_records(user_course_input)
    result["enrollments"] = _only(
        result["enrollments"],
        "start_date",
        "end_date",
        "status",
        "self_paced",
        "enrollment_mode",
        "course_url",
    )
    # From any run, since the one listed may not be the one that earned it
    certificates = {
        s["course_id"] or s["run_id"]: {
            "run_id": s["run_id"],
            "url": s["certificate_url"],
        }
        for s in summaries
        if s["certificate_earned"]
    }
    for summary in result["enrollments"]:
        summary["certificate"] = certificates.get(
            summary["course_id"] or summary["run_id"]
        )
    # From every run, since the one kept per course may not be the page's run
    page_enrollment = next(
        iter(
            _current_runs(
                [
                    s
                    for s in summaries
                    if page_resource_id in (s["course_id"], s["run_id"])
                ]
            )
        ),
        None,
    )
    if page_enrollment:
        # Small models otherwise confuse "this course" with the one in progress,
        # or an ended run with no enrollment.
        status = await _page_platform_status(platform, page_enrollment["course_id"])
        result["page_course_note"] = (
            status.note
            if status
            else "The learner is enrolled in the course on this page, "
            f"{page_enrollment['course_title']} ({page_enrollment['course_id']}); "
            f"its run is {page_enrollment['status']}. Answer a question about "
            "this course from this, not by listing their other courses."
        )
    if page_resource_id and not course_query and "page_course_note" not in result:
        # Say so, rather than leave the model to read it from a missing key. A
        # run the learner isn't in has no known course to look up a platform for.
        status = None
        platform_lookup_id = page["course_id"] or page["program_id"]
        if platform or platform_lookup_id:
            status = await _page_platform_status(platform, platform_lookup_id)
        note = status.note if status else None
        if page_resource_id.startswith(PROGRAM_ID_PREFIX):
            result["program_on_page"] = note or (
                "The learner is enrolled in the program on this page, "
                f"{page_program['program_title']} ({page_resource_id})."
                if page_program
                else "The learner has no MITx Online enrollment in "
                f"{page_resource_id}, the program on this page."
            )
        else:
            result["page_course_note"] = note or (
                f"The learner has no MITx Online enrollment in {page_resource_id}, "
                "the course or run on this page."
            )
    if "page_course_note" in result:
        result["page_resource_id"] = page_resource_id
    return json.dumps(result)


@tool(args_schema=CourseAccountToolSchema)
async def get_course_progress(
    state: Annotated[dict, InjectedState], user_course_input: str | None = None
):
    """
    Get the signed-in learner's current grade in a course (percent, letter and
    whether it is passing) and a link to the course's Progress tab, where they
    can see their progress assignment by assignment. Reads the course on this
    page, or the user_course_input given.
    """
    try:
        result = await _course_enrollments(state, user_course_input)
    except Exception:  # noqa: BLE001
        return _could_not_check("get_course_progress")
    if result["selection_status"] == AccountLookupStatus.NOT_FOUND:
        result["note"] = _not_in_records(user_course_input) + LIST_COURSES_HINT
    result["enrollments"] = _only(
        result["enrollments"], "status", "grade", "progress_url"
    )
    # Small models read a 0% grade as "nothing completed"; say it isn't.
    result["completion"] = (
        "Not available. The grade doesn't show how much of the course or how "
        "many assignments the learner has completed; link progress_url for that."
    )
    # And read a 0% grade with no letter as "not graded yet"
    result["grade_meaning"] = (
        "percent is the learner's grade so far: 0 is a grade of 0%, even with no "
        "letter. Give the grade and whether it passes, nothing more: it doesn't "
        "show what they have submitted or completed."
    )
    if any(e["grade"] is None for e in result["enrollments"]):
        result["grade_meaning"] += (
            " A grade of null means none is recorded yet: say that, not 0%."
        )
    return json.dumps(result)


# Enough for "what's due next"; the Dates tab lists the rest
MAX_DUE_DATES = 10


async def _upcoming_due_dates(run_id: str, now: datetime) -> dict:
    """
    Get the learner's next due dates in a run from its Open edX Dates tab, or
    say they couldn't be checked. Without the Open edX settings there is no
    lookup, and the Dates tab link is all the learner gets.
    """
    if not (
        settings.LEARNER_API_OPENEDX_BASE_URL
        and settings.LEARNER_API_OPENEDX_COOKIE_NAME
    ):
        return {}
    try:
        dates = await learner_api.get_course_dates(run_id)
    except Exception:  # noqa: BLE001
        log.warning("Account tool get_upcoming_assignments couldn't read due dates")
        return {"due_dates_note": "The due dates couldn't be checked."}
    upcoming = sorted(
        (
            (due, block)
            for block in dates.get("course_date_blocks") or []
            if block.get("date_type") == "assignment-due-date"
            and (due := _parse_date(block.get("date")))
            and due > now
        ),
        key=lambda item: item[0],
    )
    return {
        "upcoming_due_dates": [
            {
                "title": block.get("title"),
                "assignment_type": block.get("assignment_type"),
                "due": due.isoformat(),
                "complete": block.get("complete"),
                "learner_has_access": block.get("learner_has_access"),
                "link": block.get("link"),
            }
            for due, block in upcoming[:MAX_DUE_DATES]
        ],
        "more_due_dates": max(len(upcoming) - MAX_DUE_DATES, 0),
    }


@tool(args_schema=CourseAccountToolSchema)
async def get_upcoming_assignments(
    state: Annotated[dict, InjectedState], user_course_input: str | None = None
):
    """
    Get the signed-in learner's upcoming assignment due dates in their run of
    a course, soonest first, with the run's start and end dates and a link to
    its Dates tab. Reads the page's course, or the user_course_input given. Only
    the next few due dates are listed; more_due_dates counts the rest, which
    the Dates tab shows. With no upcoming_due_dates, or a due_dates_note saying
    they couldn't be checked, send the learner to the Dates tab.
    """
    try:
        result = await _course_enrollments(state, user_course_input)
    except Exception:  # noqa: BLE001
        return _could_not_check("get_upcoming_assignments")
    if result["selection_status"] == AccountLookupStatus.NOT_FOUND:
        result["note"] = _not_in_records(user_course_input) + LIST_COURSES_HINT
    now = datetime.now(UTC)
    due_dates = await asyncio.gather(
        *(_upcoming_due_dates(e["run_id"], now) for e in result["enrollments"])
    )
    result["enrollments"] = [
        {**summary, **dates}
        for summary, dates in zip(
            _only(
                result["enrollments"], "start_date", "end_date", "status", "dates_url"
            ),
            due_dates,
        )
    ]
    return json.dumps(result)


def _program_certificate_summary(certificate: dict) -> dict:
    """Keep a program certificate's title, date and link; drop personal data."""
    return {
        "program_title": certificate.get("program_title"),
        "completed_on": certificate.get("program_completion_timestamp"),
        "certificate_url": certificate.get("program_letter_share_url"),
    }


def _program_enrollment_summary(enrollment: dict) -> dict:
    """Keep a program enrollment's title and whether it earned a certificate."""
    program = enrollment.get("program") or {}
    certificate = enrollment.get("certificate")
    return {
        "program_title": program.get("title"),
        "program_id": program.get("readable_id"),
        "certificate_earned": bool(certificate),
        "certificate_url": _certificate_url(certificate),
        "certificate_uuid": _certificate_uuid(certificate),
    }


async def _digital_credential(get_certificate, summary: dict) -> str | None:
    """
    Return "issued" only when MITx Online returns the signed credential, and
    None when there is no certificate. A missing credential can mean the course
    doesn't issue one or that issuing it failed (hq#13746), so anything else is
    "not_confirmed", never "none".
    """
    if not summary["certificate_earned"]:
        return None
    try:
        certificate = await get_certificate(summary["certificate_uuid"])
    except learner_api.LearnerAPIError:
        return DIGITAL_CREDENTIAL_NOT_CONFIRMED
    if (
        isinstance(certificate, dict)
        and certificate.get("verifiable_credential_json")
        and not certificate.get("is_revoked")
    ):
        return DIGITAL_CREDENTIAL_ISSUED
    return DIGITAL_CREDENTIAL_NOT_CONFIRMED


@tool(args_schema=CertificateToolSchema)
async def get_certificate_status(  # noqa: C901, PLR0912
    state: Annotated[dict, InjectedState],
    user_course_input: str | None = None,
    all_courses: bool = False,  # noqa: FBT001, FBT002
):
    """
    Get whether the signed-in learner has earned a certificate in a course,
    with its link, plus their program enrollments and program certificates.
    A program certificate can show up in either place: as certificate_earned
    on a program, or in program_certificates (MIT Learn's program records).
    Use it for "did I get a certificate", "where is my certificate", "where is
    my digital credential" or "have I finished the program". The course
    certificate is for the page's course, or the user_course_input given. For
    "what certificates do I have", pass all_courses to list all of the
    learner's courses.

    Each earned MITx Online certificate also has digital_credential, for the
    separate digital credential: "issued" when one exists, "not_confirmed" when
    it couldn't be confirmed (send the learner to support for that one).
    """
    resource_query = _normalize_course_input(user_course_input)
    try:
        if all_courses:
            result = {
                "enrollments": _current_runs(
                    await _summaries(), prefer_certificate=True
                )
            }
            page, platform = _get_page_context(state)
            if page["resource_id"] and platform and platform != MITXONLINE_PLATFORM:
                # So a question about this page isn't answered from these
                result["note"] = (
                    "These are the learner's MITx Online records only. The "
                    "course or program on this page isn't on MITx Online, so "
                    "its certificate can't be checked here; don't answer a "
                    "question about it from these."
                )
        else:
            result = await _course_enrollments(
                state, user_course_input, prefer_certificate=True
            )
        if result.get("selection_status") == AccountLookupStatus.NO_PAGE_COURSE:
            result["note"] += " For all of their certificates, pass all_courses."
            return json.dumps(result)
        if result.get("selection_status") in (
            AccountLookupStatus.NOT_ON_MITXONLINE,
            AccountLookupStatus.PLATFORM_UNCLEAR,
        ):
            # No course to answer for, or another platform's, so the learner's
            # MITx Online programs aren't read either, or a model answers from them
            return json.dumps(result)
        programs, program_certificates = await asyncio.gather(
            learner_api.get_program_enrollments(),
            learner_api.get_program_certificates(),
        )
    except Exception:  # noqa: BLE001
        return _could_not_check("get_certificate_status")
    enrollments = result["enrollments"]
    programs = [_program_enrollment_summary(p) for p in programs]
    if (
        resource_query
        and not all_courses
        and (
            status := await _page_program_status(
                state, _named_programs(resource_query, programs)
            )
        )
    ):
        return json.dumps(_empty_course_selection(status))
    credentials = await asyncio.gather(
        *(
            _digital_credential(learner_api.get_course_certificate, summary)
            for summary in enrollments
        ),
        *(
            _digital_credential(learner_api.get_program_certificate, summary)
            for summary in programs
        ),
    )
    result["enrollments"] = _only(
        enrollments,
        "status",
        "enrollment_mode",
        "certificate_earned",
        "certificate_url",
    )
    for summary in programs:
        del summary["certificate_uuid"]
    result["programs"] = programs
    for summary, credential in zip(
        [*result["enrollments"], *programs], credentials, strict=True
    ):
        if credential:
            summary["digital_credential"] = credential
    if DIGITAL_CREDENTIAL_NOT_CONFIRMED in credentials:
        # Small models read "not_confirmed" as "not issued"
        result["digital_credential_meaning"] = (
            "not_confirmed means it couldn't be confirmed whether a digital "
            "credential was issued. Don't say it wasn't issued or doesn't exist; "
            "send the learner to support for it."
        )
    result["program_certificates"] = [
        _program_certificate_summary(c) for c in program_certificates
    ]
    if result.get("selection_status") == AccountLookupStatus.NOT_FOUND:
        # A program's title isn't a course; don't say it isn't in their records.
        # MIT Learn can hold a program certificate with no MITx Online enrollment.
        named = {
            p["program_title"] for p in _named_programs(resource_query, programs)
        } | {
            c["program_title"]
            for c in result["program_certificates"]
            if _matches_resource_query(resource_query, c)
        }
        if named:
            result["selection_status"] = AccountLookupStatus.MATCHED
            result["note"] = (
                f"{user_course_input} is the learner's program "
                f"{', '.join(sorted(named))}; "
                "its entry in programs or program_certificates says whether they "
                "earned its certificate."
            )
        else:
            result["note"] = _not_in_records(user_course_input) + LIST_COURSES_HINT
    if result.get("selection_status") == AccountLookupStatus.ON_PROGRAM_PAGE:
        # Only set once the page's program is known to be MITx Online's
        page_program = _get_page_context(state)[0]["program_id"]
        result["program_on_page"] = next(
            (
                f"The program on this page is {p['program_title']} "
                f"({p['program_id']}); its entry in programs says whether the "
                "learner earned its certificate."
                for p in programs
                if p["program_id"] == page_program
            ),
            "The learner has no MITx Online enrollment in the program on this "
            "page. program_certificates can still list its certificate, by title.",
        )
    return json.dumps(result)


ACCOUNT_TOOLS = [
    get_enrollments,
    get_course_progress,
    get_certificate_status,
    get_upcoming_assignments,
]
