"""Tools and schemas for AI agents"""

import json
import logging
from typing import Annotated

import pydantic
from bs4 import BeautifulSoup
from django.conf import settings
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from ai_chatbots import learn
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


def _envelope(result: learn.SearchResult, shaped: list[dict], **extra) -> str:
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
        description="The agent state, including the search url to use"
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
    backend = ((state or {}).get("search_backend") or ["vector"])[-1]
    try:
        result = await learn.search_learning_resources(q, filters, backend=backend)
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
        result = await learn.search_content_files(
            q, readable_ids, collection_name=collection_name, platform=platform
        )
        # Consumers store the flag as a string, evaluation sets as a bool
        exclude = str(exclude_canvas) != "False"
        shaped = []
        citations = {}
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
        return _envelope(result, shaped, citation_sources=citations)
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
        result = await learn.search_transcript(q, state["transcript_asset_id"][-1])
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
        results = await learn.learning_resources(readable_id)
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
