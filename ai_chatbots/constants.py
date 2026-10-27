"""Constants for the AI Chat application."""

import dataclasses
import datetime
import re

from django.conf import settings
from named_enum import ExtendedEnum

HYBRID_SEARCH_FEATURE_FLAG = "bot_hybrid_search_enabled"
# MIT Learn search filters a client can pin for a recommendation thread,
# e.g. a channel page's topic. The search tool always applies them.
RECOMMENDATION_SEARCH_FILTERS = (
    "certification",
    "certification_type",
    "course_feature",
    "delivery",
    "department",
    "free",
    "level",
    "ocw_topic",
    "offered_by",
    "platform",
    "professional",
    "resource_category",
    "resource_type",
    "resource_type_group",
    "topic",
)
GROUP_STAFF_AI_SYTEM_PROMPT_EDITORS = "ai_system_prompt_editors"
AI_ANONYMOUS_USER = "anonymous"
AI_THREAD_COOKIE_KEY = "ai_thread_auth"
AI_THREADS_ANONYMOUS_COOKIE_KEY = "ai_threads_anon"
AI_SESSION_COOKIE_KEY = "ai_odl_unique_id"
AZURE_MODEL_PREFIX = "azure/"
AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


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

# Zendesk ticket creation endpoint, appended to the API base url
ZENDESK_TICKET_PATH = "/api/v2/tickets.json"

# Zendesk OAuth token endpoint, appended to the API base url. Used to mint a
# client_credentials access token for ticket creation.
ZENDESK_OAUTH_TOKEN_PATH = "/oauth/tokens"  # noqa: S105

# Tag applied to every ticket filed by the support intake widget, so these can be
# told apart from tickets raised through the help center or by email.
ZENDESK_SUPPORT_INTAKE_TAG = "tim_support_intake"

# Added when the requester address was asserted by the caller rather than taken
# from an authenticated session, so support does not read it as confirmed.
ZENDESK_UNVERIFIED_REQUESTER_TAG = "tim_unverified_requester"

# What the requester was on, derived from the User-Agent. A controlled
# vocabulary, because tags are what agents filter views on; the verbatim user
# agent goes in the ticket body for anything finer. Nothing is tagged when the
# header is missing or ambiguous - a wrong device is worse than no device.
ZENDESK_DEVICE_MOBILE_TAG = "tim_device_mobile"
ZENDESK_DEVICE_TABLET_TAG = "tim_device_tablet"
ZENDESK_DEVICE_DESKTOP_TAG = "tim_device_desktop"

# Stands in for the learner's own words on an unverified ticket: a
# received-request trigger puts the subject straight into an email to an address
# nobody has proved belongs to whoever typed it.
ZENDESK_UNVERIFIED_SUBJECT = "Support request from MIT Open Learning"

# Zendesk help center category ids, as listed by the public category endpoint:
#   curl -sL https://support.learn.mit.edu/api/v2/help_center/categories.json
#     41249004008859  About MIT Learn
#     41410514373019  Universal Learning
#     41750628505627  MITx
#     41249375945243  MIT xPRO
#     41249707771035  MIT OpenCourseWare (OCW)

# Mapping from platform to zendesk category id
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

# Universal AI (UAI) courses and programs are hosted on MITx Online, so their
# platform code is "mitxonline", but their support articles live in the
# "Universal Learning" category instead of the MITx one.  Only the readable id
# distinguishes them, and it comes in two shapes: program-v1:UAI+B2C* for the
# programs and course-v1:UAI_SOURCE+UAI.* for the courses.
ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID = "41410514373019"
UAI_READABLE_ID_REGEX = re.compile(r"^(?:course|program)-v1:UAI(?:_\w+)?\+")


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
        cross_site = (
            "SameSite=None;Secure;" if settings.AI_CHATBOTS_COOKIE_CROSS_SITE else ""
        )
        return f"{self.name}={self.value};Path={self.path};{expire_str};{cross_site}"


WRITES_MAPPING = {"human": "__start__", "ai": "agent", "tool": "tools"}
