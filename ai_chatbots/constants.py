"""Constants for the AI Chat application."""

from langchain_openai import ChatOpenAI
from named_enum import ExtendedEnum

GROUP_STAFF_AI_SYTEM_PROMPT_EDITORS = "ai_system_prompt_editors"
AI_ANONYMOUS_USER = "anonymous"
AI_THREAD_COOKIE_KEY = "ai_thread_id"


class LLMClassEnum(ExtendedEnum):
    """
    Enum for determining which LLM class to
    use based on settings.AI_PROVIDER. For example,
    if AI_PROVIDER == "openai", the OpenAI LLM class
    should be used.
    """

    openai = ChatOpenAI


class LearningResourceType(ExtendedEnum):
    """Enum for LearningResource resource_type values"""

    course = "Course"
    program = "Program"
    learning_path = "Learning Path"
    podcast = "Podcast"
    podcast_episode = "Podcast Episode"
    video = "Video"
    video_playlist = "Video Playlist"


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
