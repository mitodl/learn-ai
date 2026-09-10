"""Learner context: stated profile (from MIT Learn) plus memory learned from chats."""

import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from langchain.chat_models import init_chat_model
from langmem import create_memory_store_manager
from pydantic import BaseModel, Field

from ai_chatbots.memorystores import DjangoMemoryStore
from main.features import is_enabled

log = logging.getLogger(__name__)

CHAT_MEMORY_FLAG = "CHAT_MEMORY"
MEMORY_KEY = "default"  # langmem's fixed key for a single-document profile
# Wording matters: a polite "use this" version still lost to the static prompt's
# "ask clarifying questions" examples with gpt-4o-mini; this one wins while appended.
CONTEXT_INSTRUCTION = (
    "# Learner context\n"
    "The learner has ALREADY answered the clarifying questions below through their "
    "profile. Treat these as their answers and do not ask them again. Search now using "
    "them; only ask about things not listed here."
)
# Without this, memory changed nothing downstream: "machine learning" in current focus
# still gave search_courses(q="data science") and social-science results despite avoid.
SEARCH_INSTRUCTION = (
    "When you search, put the current focus in the query and, after searching, leave "
    "out any result about a topic listed under Avoid."
)
PROFILE_HEADING = "## About this learner (stated in their MIT Learn profile)"
MEMORY_HEADING = "## Learned from prior chats"
# Certificate tracks are the paid ones at MIT Learn, so this field also answers price.
CERTIFICATE_TEXT = {
    "yes": "wants one (paid certificate courses are fine)",
    "no": "does not want one (free, non-certificate resources are fine; "
    "no need to ask about price)",
}


class LearnerMemory(BaseModel):
    """What the chatbots learned about a learner; one document, revised over time."""

    background: str = Field("", description="Education, occupation, prior knowledge")
    goals_and_interests: str = Field("", description="What they want to learn and why")
    learning_preferences: str = Field(
        "", description="Format, pace, delivery and level preferences"
    )
    current_focus: str = Field("", description="What they are working on right now")
    avoid: str = Field(
        "", description="Topics, fields or resource types they asked not to be shown"
    )


EXTRACTION_INSTRUCTIONS = """You maintain a short profile of a learner using MIT Open
Learning chatbots. Update it from the conversation: keep it factual, about the learner
only, and under 1500 characters in total. You only ever see the learner's own
messages. Record preferences, background, goals, current focus, and anything they
asked not to be recommended (put that under avoid). A bare question with no
personal detail says nothing about the learner: then change nothing. Never record
names, emails, quiz or problem answers, grades, or problem identifiers. Drop anything
the learner contradicts."""


def memory_namespace(global_id: str) -> tuple[str, str]:
    return ("memories", global_id)


def fetch_learner_profile(global_id: str) -> dict:  # noqa: ARG001
    """
    ponytail: dummy stand-in for GET /api/v0/profiles/<global_id>/preferences/ on
    mit-learn (see docs/rfc-learner-memory.md). Replace with async_request plus a
    12h cache keyed on global_id.
    """
    return {
        "topic_interests": [
            {
                "id": 731,
                "name": "Science & Math",
                "icon": "RiTestTubeLine",
                "parent": None,
                "channel_url": "https://learn.mit.edu/c/topic/science-math",
            }
        ],
        "goals": ["academic-excellence"],
        "current_education": "",
        "certificate_desired": "no",
        "time_commitment": "",
        "delivery": ["online"],
        "preference_search_filters": {
            "certification": False,
            "topic": ["Science & Math"],
            "delivery": ["online"],
        },
    }


def memory_enabled(user) -> bool:
    """Return True for a logged-in learner with the CHAT_MEMORY flag on."""
    if not isinstance(user, get_user_model()) or not user.global_id:
        return False
    return is_enabled(CHAT_MEMORY_FLAG, default=False, opt_unique_id=user.global_id)


def get_learner_memory(global_id: str) -> dict | None:
    item = DjangoMemoryStore().get(memory_namespace(global_id), MEMORY_KEY)
    return item.value.get("content") if item else None


def build_learner_context(profile: dict, memory: dict | None) -> str:
    """Render the prompt block; blank fields are skipped, empty input gives ''."""
    lines = []
    profile_rows = [
        ("Topics of interest", [t["name"] for t in profile.get("topic_interests", [])]),
        ("Goals", profile.get("goals")),
        ("Current education", profile.get("current_education")),
        ("Certificate", CERTIFICATE_TEXT.get(profile.get("certificate_desired"))),
        ("Time commitment", profile.get("time_commitment")),
        ("Preferred delivery", profile.get("delivery")),
    ]
    profile_lines = [
        f"- {label}: {', '.join(v) if isinstance(v, list) else v}"
        for label, v in profile_rows
        if v
    ]
    if profile_lines:
        lines += [CONTEXT_INSTRUCTION, PROFILE_HEADING, *profile_lines]
    if memory:
        memory_lines = [
            f"- {field.replace('_', ' ').capitalize()}: {text}"
            for field, text in memory.items()
            if text
        ]
        if memory_lines:
            body = "\n".join(memory_lines)[: settings.AI_MEMORY_MAX_CHARS]
            if not lines:
                lines.append(CONTEXT_INSTRUCTION)
            lines += [MEMORY_HEADING, body, SEARCH_INSTRUCTION]
    return "\n".join(lines)


def get_learner_context(user) -> str:
    """Return the learner context block, or '' if disabled or anonymous."""
    if not memory_enabled(user):
        return ""
    try:
        return build_learner_context(
            fetch_learner_profile(user.global_id), get_learner_memory(user.global_id)
        )
    except Exception:
        log.exception("Learner context unavailable for %s", user.global_id)
        return ""


def get_memory_manager():
    """
    Return the langmem store manager that revises the single LearnerMemory document.

    ponytail: init_chat_model, not ChatLiteLLM. trustcall's patch loop never converges
    with ChatLiteLLM (GraphRecursionError, observed with gpt-4o-mini), so the extraction
    model bypasses the LiteLLM proxy for now.
    """
    return create_memory_store_manager(
        init_chat_model(settings.AI_MEMORY_EXTRACTION_MODEL, temperature=0),
        schemas=[LearnerMemory],
        instructions=EXTRACTION_INSTRUCTIONS,
        default_factory=lambda _config: LearnerMemory(),
        enable_inserts=False,
        namespace=("memories", "{langgraph_user_id}"),
        store=DjangoMemoryStore(),
    )
