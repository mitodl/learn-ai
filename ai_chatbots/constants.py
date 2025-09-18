"""Constants for the AI Chat application."""

import dataclasses
import datetime
from typing import Optional

from named_enum import ExtendedEnum

GROUP_STAFF_AI_SYTEM_PROMPT_EDITORS = "ai_system_prompt_editors"
AI_ANONYMOUS_USER = "anonymous"
AI_THREAD_COOKIE_KEY = "ai_thread_auth"
AI_THREADS_ANONYMOUS_COOKIE_KEY = "ai_threads_anon"


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


@dataclasses.dataclass
class ChatbotCookie:
    name: str
    value: str
    path: str = "/"
    max_age: Optional[datetime.datetime] = None

    def __str__(self) -> str:
        """
        Represent the cookie as a string
        """
        expire_str = f"Max-Age={self.max_age}" if self.max_age is not None else ""
        return f"{self.name}={self.value};Path={self.path};{expire_str};"


WRITES_MAPPING = {"human": "__start__", "ai": "agent", "tool": "tools"}
