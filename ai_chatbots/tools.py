"""Tools and schemas for AI agents"""

import json
import logging
from typing import Annotated, Optional

import pydantic
import requests
from django.conf import settings
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from ai_chatbots.constants import LearningResourceType, OfferedBy
from ai_chatbots.utils import enum_zip

log = logging.getLogger(__name__)


class SearchToolSchema(pydantic.BaseModel):
    """Schema for searching MIT learning resources.

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
    resource_type: Optional[list[enum_zip("resource_type", LearningResourceType)]] = (
        Field(
            default=None,
            description=(
                """
                Type of resource to search for: course, program, video, etc.
                If the user mentions courses, programs, videos, or podcasts in
                particular, filter the search by this parameter.  DO NOT USE THE
                resource_typeFILTER OTHERWISE. You MUST combine multiple resource types
                in one request like this: "resource_type=course&resource_type=program".
                Do not attempt more than one query peruser message. If the user asks for
                podcasts, filter by both "podcast" and "podcast_episode".
                """
            ),
        )
    )
    free: Optional[bool] = Field(
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
    certification: Optional[bool] = Field(
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
    offered_by: Optional[list[enum_zip("resource_type", OfferedBy)]] = Field(
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

    state: Annotated[dict, InjectedState] = Field(
        description="The agent state, including the search url to use"
    )


@tool(args_schema=SearchToolSchema)
def search_courses(q: str, state: Annotated[dict, InjectedState], **kwargs) -> str:
    """
    Query the MIT API for learning resources, and
    return simplified results as a JSON string
    """

    params = {"q": q, "limit": settings.AI_MIT_SEARCH_LIMIT}

    valid_params = {
        "resource_type": [rt.name for rt in kwargs.get("resource_type", [])] or None,
        "free": kwargs.get("free"),
        "offered_by": [o.name for o in kwargs.get("offered_by", [])] or None,
        "certification": kwargs.get("certification"),
    }
    params.update({k: v for k, v in valid_params.items() if v is not None})
    search_url = state["search_url"][-1]
    log.debug("Searching MIT API at %s with params: %s", search_url, params)
    try:
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        raw_results = response.json().get("results", [])
        # Simplify the response to only include the main properties
        main_properties = [
            "title",
            "id",
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
    except requests.exceptions.RequestException:
        log.exception("Error querying MIT API")
        return json.dumps({"error": "An error occurred while searching"})


class SearchContentFilesToolSchema(pydantic.BaseModel):
    """Schema for searching MIT contentfiles related to a particular course."""

    q: str = Field(
        description=(
            "Query to find course information that might answer the user's question."
        )
    )
    state: Annotated[dict, InjectedState] = Field(
        description="The agent state, including course_id and collection_name params"
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


@tool(args_schema=SearchContentFilesToolSchema)
def search_content_files(q: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Query the MIT contentfile vector endpoint API, and return results as a
    JSON string, along with metadata about the query parameters used.
    """

    url = settings.AI_MIT_SYLLABUS_URL
    course_id = state["course_id"][-1]
    collection_name = state["collection_name"][-1]
    params = {
        "q": q,
        "resource_readable_id": course_id,
        "limit": settings.AI_MIT_CONTENT_SEARCH_LIMIT,
    }
    if collection_name:
        params["collection_name"] = collection_name
    log.info("Searching MIT API with params: %s", params)
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        raw_results = response.json().get("results", [])
        # Simplify the response to only include the main properties
        simplified_results = []
        for result in raw_results:
            simplified_result = {
                "chunk_content": result.get("chunk_content"),
                "run_title": result.get("run_title"),
            }
            simplified_results.append(simplified_result)
        full_output = {
            "results": simplified_results,
            "metadata": {"parameters": params},
        }
        return json.dumps(full_output)
    except requests.exceptions.RequestException:
        log.exception("Error querying MIT API")
        return json.dumps({"error": "An error occurred while searching"})


@tool(args_schema=VideoGPTToolSchema)
def get_video_transcript_chunk(q: str, state: Annotated[dict, InjectedState]) -> str:
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

    log.info("Searching MIT API with params: %s", params)
    try:
        response = requests.get(url, params=params, timeout=30)
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
    except requests.exceptions.RequestException:
        log.exception("Error querying MIT API for transcripts")
        return json.dumps({"error": "An error occurred while getting the transcript"})
