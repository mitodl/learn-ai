"""Constants for the AI Chat application."""

import dataclasses
import datetime

from named_enum import ExtendedEnum

HYBRID_SEARCH_FEATURE_FLAG = "bot_hybrid_search_enabled"
GROUP_STAFF_AI_SYTEM_PROMPT_EDITORS = "ai_system_prompt_editors"
AI_ANONYMOUS_USER = "anonymous"
AI_THREAD_COOKIE_KEY = "ai_thread_auth"
AI_THREADS_ANONYMOUS_COOKIE_KEY = "ai_threads_anon"
AI_SESSION_COOKIE_KEY = "ai_odl_unique_id"


class LearningResourceType(ExtendedEnum):
    """Enum for LearningResource resource_type values"""

    course = "Course"
    program = "Program"
    learning_path = "Learning Path"
    podcast = "Podcast"
    podcast_episode = "Podcast Episode"
    video = "Video"
    video_playlist = "Video Playlist"
    document = "Document"


class OfferedBy(ExtendedEnum):
    """
    Enum for our Offered By labels. They are our MIT "brands" for LearningResources
    (Courses, Bootcamps, Programs) and are independent of what platform.
    User generated lists UserLists (like a learning path) don't have offered by "brand".
    Values are user-facing.
    These should be kept in sync with the LearningResourceOfferor model objects
    """

    mitx = "MITx"
    ocw = "MIT OpenCourseWare"
    bootcamps = "Bootcamps"
    xpro = "MIT xPRO"
    mitpe = "MIT Professional Education"
    see = "MIT Sloan Executive Education"


# Zendesk help center article search endpoint, appended to the help center base url
ZENDESK_ARTICLE_SEARCH_PATH = "/api/v2/help_center/articles/search.json"

# The MIT Learn help center is divided into one category per MIT platform, so a
# support search has to be limited to the category matching the platform of the
# course under discussion.  Otherwise keyword relevance alone decides, and i.e.
# "certificate" asked about an OCW course is answered with an MIT xPRO article.
# Each platform category repeats the articles it needs (account creation,
# technical issues, getting support), so searching one category is enough.
# Keys are MIT Learn platform codes, values are MIT Learn help center category ids.
ZENDESK_PLATFORM_CATEGORY_IDS = {
    # MIT OpenCourseWare (OCW)
    "ocw": "41249707771035",
    # MITx, which covers the MITx courses now hosted on MIT Learn / MITx Online
    "mitxonline": "41750628505627",
    "edx": "41750628505627",
    # MIT xPRO, including its Emeritus, Global Alumni and WHU partner courses,
    # which each have their own section under the xPRO category
    "xpro": "41249375945243",
    "emeritus": "41249375945243",
    "globalalumni": "41249375945243",
    "whu": "41249375945243",
}


class ChatResponseScore(ExtendedEnum):
    """Enum for chat response ratings"""

    like = "like"
    dislike = "dislike"
    no_rating = ""


@dataclasses.dataclass
class ChatbotCookie:
    name: str
    value: str
    path: str = "/"
    max_age: datetime.datetime | None = None

    def __str__(self) -> str:
        """
        Represent the cookie as a string
        """
        expire_str = f"Max-Age={self.max_age}" if self.max_age is not None else ""
        return f"{self.name}={self.value};Path={self.path};{expire_str};"


WRITES_MAPPING = {"human": "__start__", "ai": "agent", "tool": "tools"}
