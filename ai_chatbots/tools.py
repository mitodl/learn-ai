"""Tools and schemas for AI agents"""

import json
import logging
from typing import Annotated

import pydantic
from asgiref.sync import sync_to_async
from bs4 import BeautifulSoup
from django.conf import settings
from httpx import HTTPStatusError, RequestError
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from ai_chatbots.constants import (
    HYBRID_SEARCH_FEATURE_FLAG,
    UAI_READABLE_ID_REGEX,
    ZENDESK_ARTICLE_SEARCH_PATH,
    ZENDESK_PLATFORM_CATEGORY_IDS,
    ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID,
    LearningResourceType,
    OfferedBy,
)
from ai_chatbots.utils import async_request, enum_zip, get_django_cache
from main.features import is_enabled as feature_is_enabled

log = logging.getLogger(__name__)

# Cache key prefix for the platform a course is offered on
COURSE_PLATFORM_CACHE_PREFIX = "course_platform_"


async def _is_hybrid_search_enabled() -> bool:
    """Check if the hybrid search feature flag is enabled."""
    return await sync_to_async(feature_is_enabled)(
        HYBRID_SEARCH_FEATURE_FLAG, default=False
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
    offered_by: list[enum_zip("resource_type", OfferedBy)] | None = Field(
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
    params = {"q": q, "limit": settings.AI_MIT_SEARCH_LIMIT}

    valid_params = {
        "resource_type": [rt.name for rt in (kwargs.get("resource_type") or [])]
        or None,
        "free": kwargs.get("free"),
        "offered_by": [o.name for o in (kwargs.get("offered_by") or [])] or None,
        "certification": kwargs.get("certification"),
    }
    if await _is_hybrid_search_enabled():
        valid_params["hybrid_search"] = True

    params.update({k: v for k, v in valid_params.items() if v is not None})
    search_url = state["search_url"][-1] if state else settings.AI_MIT_SEARCH_URL
    log.debug("Searching MIT resources API at %s with params: %s", search_url, params)
    try:
        response = await async_request(
            search_url,
            params,
            timeout=settings.REQUESTS_TIMEOUT,
            include_learn_token=True,
        )
        response.raise_for_status()
        raw_results = response.json().get("results", [])
        # Ignore any results over the maximum limit
        raw_results = raw_results[: settings.AI_MIT_SEARCH_LIMIT]
        # Simplify the response to only include the main properties
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
        simplified_results = []
        for result in raw_results:
            simplified_result = {k: result.get(k) for k in main_properties}
            simplified_result["url"] = (
                f"{settings.AI_MIT_SEARCH_DETAIL_URL}{result.pop('id')}"
            )
            # Instructors and level will be in the runs data if present
            next_date = result.get("next_start_date", None)
            raw_runs = result.get("runs", [])
            best_run = None
            if next_date:
                runs = [run for run in raw_runs if run["start_date"] == next_date]
                if runs:
                    best_run = runs[0]
            elif raw_runs:
                best_run = raw_runs[-1]
            if best_run:
                for attribute in ("level", "instructors"):
                    simplified_result[attribute] = best_run.get(attribute, [])
            simplified_results.append(simplified_result)
        full_output = {
            "results": simplified_results,
            "metadata": {"search_url": search_url, "parameters": params},
        }
        return json.dumps(full_output)
    except (RequestError, HTTPStatusError):
        log.exception("Error querying MIT API")
        return json.dumps({"error": "An error occurred while searching"})


class SearchContentFilesToolSchema(pydantic.BaseModel):
    """
    Schema for searching MIT contentfiles related to a particular learning resource.
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


async def _content_file_search(url, params, *, exclude_canvas=True):
    try:
        # Convert the exclude_canvas parameter to a boolean if it is a string
        if exclude_canvas and exclude_canvas == "False":
            exclude_canvas = False
        if await _is_hybrid_search_enabled():
            params["hybrid_search"] = True

        response = await async_request(
            url, params, timeout=settings.REQUESTS_TIMEOUT, include_learn_token=True
        )
        response.raise_for_status()
        raw_results = response.json().get("results", [])
        # Simplify the response to only include the main properties
        simplified_results = []
        citations = {}
        for result in raw_results:
            platform = result.get("platform", {}).get("code")
            # Currently, canvas contentfiles have blank platform values,
            # those from other sources do not.
            if exclude_canvas and (not platform or platform == "canvas"):
                continue
            simplified_result = {
                "id": result["resource_point_id"],
                "chunk_content": result.get("chunk_content"),
                "run_title": result.get("run_title"),
            }
            simplified_results.append(simplified_result)
            if result.get("url") and not citations.get(result["resource_point_id"]):
                citations[result["resource_point_id"]] = {
                    "citation_url": result.get("url"),
                    "citation_title": result.get("title")
                    or result.get("content_title"),
                }
        full_output = {
            "results": simplified_results,
            "citation_sources": citations,
            "metadata": {"parameters": params},
        }
        return json.dumps(full_output)
    except Exception:
        log.exception("Error querying MIT API")
        return json.dumps({"error": "An error occurred while searching"})


@tool(args_schema=SearchContentFilesToolSchema)
async def search_content_files(
    q: str, state: Annotated[dict, InjectedState], readable_id: str | None = None
) -> str:
    """
    Search for detailed information about a particular MIT learning resource.
    The resource is identified by its readable_id or course_id.
    """

    url = settings.AI_MIT_SYLLABUS_URL
    course_id = state.get("course_id", [None])[-1] or readable_id
    collection_name = state.get("collection_name", [None])[-1]
    exclude_canvas = state.get("exclude_canvas", ["True"])[-1]
    params = {
        "q": q,
        "resource_readable_id": course_id,
        "limit": settings.AI_MIT_CONTENT_SEARCH_LIMIT,
    }
    if collection_name:
        params["collection_name"] = collection_name
    return await _content_file_search(url, params, exclude_canvas=exclude_canvas)


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
    url = settings.AI_MIT_SYLLABUS_URL
    collection_name = state.get("collection_name", [None])[-1]
    related_courses = state["related_courses"]
    params = {
        "q": q,
        "resource_readable_id": related_courses,
        "limit": settings.AI_MIT_CONTENT_SEARCH_LIMIT,
    }
    if collection_name:
        params["collection_name"] = collection_name
    return await _content_file_search(url, params)


@tool(args_schema=VideoGPTToolSchema)
async def get_video_transcript_chunk(
    q: str, state: Annotated[dict, InjectedState]
) -> str:
    """
    Query the MIT video transcript API, and return results as a JSON string.
    """

    url = settings.AI_MIT_VIDEO_TRANSCRIPT_URL

    transcript_asset_id = state["transcript_asset_id"][-1]
    params = {
        "q": q,
        "edx_module_id": transcript_asset_id,
        "limit": settings.AI_MIT_TRANSCRIPT_SEARCH_LIMIT,
    }

    log.debug("Searching MIT API with params: %s", params)
    try:
        response = await async_request(
            url, params, timeout=settings.REQUESTS_TIMEOUT, include_learn_token=True
        )
        response.raise_for_status()
        raw_results = response.json().get("results", [])
        # Simplify the response to only include the main properties
        simplified_results = []
        for result in raw_results:
            simplified_result = {
                "chunk_content": result.get("chunk_content"),
            }
            simplified_results.append(simplified_result)
        full_output = {
            "results": simplified_results,
            "metadata": {"parameters": params},
        }

        return json.dumps(full_output)
    except Exception:
        log.exception("Error querying MIT API for transcripts")
        return json.dumps({"error": "An error occurred while getting the transcript"})


class SearchSupportArticlesToolSchema(pydantic.BaseModel):
    """
    Schema to search the MIT Learn support center (Zendesk help center) for
    articles about how MIT platforms work: enrollment, certificates, refunds,
    payments, account and login issues, deadlines, technical problems, etc.

    Here are some recommended tool parameters to apply for sample user prompts:

    User: "How do I get a certificate for my course?"
    Search parameters: q="certificate"

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
            """
        )
    )

    state: Annotated[dict, InjectedState] = Field(
        description=(
            "The agent state, which may include the course_id of the resource "
            "under discussion.  Its platform determines which part of the "
            "support center is searched."
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
        response = await async_request(
            settings.AI_MIT_LEARNING_RESOURCES_URL,
            {"readable_id": readable_id, "limit": 1},
            # A cheap metadata lookup, not a search, and every support search
            # waits on it, so it gets a tighter timeout than the default.
            timeout=settings.AI_COURSE_PLATFORM_LOOKUP_TIMEOUT,
            # Courses are public, so the lookup still works without a token.
            # httpx rejects an empty bearer header outright, so only send one
            # when a token is actually configured.
            include_learn_token=bool(settings.LEARN_ACCESS_TOKEN),
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        platform = (results[0].get("platform") or {}).get("code") if results else None
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


@tool(args_schema=SearchSupportArticlesToolSchema)
async def search_support_articles(q: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Search the MIT Learn support center (help center) for up to date articles
    about how MIT platforms and courses work: enrollment, certificates,
    refunds, payments, accounts, logins, deadlines and technical issues, and
    also how course material is delivered - video transcripts and captions,
    accessibility, course formats, prerequisites, and how long access to the
    content lasts.  Use this tool for any question about a course that is not
    answered by the course content itself.  Returns the articles as a JSON
    string.
    """
    portal_url = settings.AI_ZENDESK_URL
    if not portal_url:
        log.warning("No support portal url is configured")
        return json.dumps({"results": []})

    search_url = f"{portal_url.rstrip('/')}{ZENDESK_ARTICLE_SEARCH_PATH}"
    params = {"query": q, "per_page": settings.AI_ZENDESK_SEARCH_LIMIT}

    # Limit the search to the parts of the support center covering the course
    # under discussion.
    course_ids = (state or {}).get("course_id") or [None]
    course_id = course_ids[-1]
    platform = await _get_course_platform(course_id) if course_id else None
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
        return json.dumps({"results": []})

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

    full_output = {
        "results": articles,
        "citation_sources": {
            article["id"]: {
                "citation_url": article["url"],
                "citation_title": article["title"],
            }
            for article in articles
            if article["url"]
        },
        "metadata": {
            "search_url": search_url,
            "parameters": params,
            "platform": platform,
        },
    }
    return json.dumps(full_output)
