"""Learner context: stated profile (from MIT Learn) plus memory learned from chats."""

import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ai_chatbots.memorystores import DjangoMemoryStore
from main.features import is_enabled

log = logging.getLogger(__name__)

CHAT_MEMORY_FLAG = "CHAT_MEMORY"
ABOUT_KEY = "about"
INSTRUCTIONS_KEY = "instructions"
# Only the recommendation bot can act on topics, certificate and delivery preferences
FULL_PROFILE_BOTS = {"ResourceRecommendationBot"}

CONTEXT_INSTRUCTION = (
    "# Learner context\n"
    "What follows is known about this learner from their MIT Learn profile and earlier "
    "chats. Treat it as their answers: do not ask for anything stated here, and follow "
    "the instructions under 'How this learner wants to be helped'."
)
ABOUT_HEADING = "## About this learner"
INSTRUCTIONS_HEADING = "## How this learner wants to be helped"
TUTOR_RULES_WIN = (
    "These learner preferences shape tone and format only; your tutoring rules always "
    "take precedence over them."
)
# Certificate tracks are the paid ones at MIT Learn, so this field also answers price.
CERTIFICATE_TEXT = {
    "yes": "wants one (paid certificate courses are fine)",
    "no": "does not want one (free, non-certificate resources are fine; "
    "no need to ask about price)",
}


class LearnerMemory(BaseModel):
    """Free-form memory: shared facts, shared instructions, one bot's instructions."""

    about: str = Field(
        "",
        description="Durable facts about the learner as a person: background, "
        "occupation, education level, goals, constraints such as available time.",
    )
    instructions: str = Field(
        "",
        description="How the learner wants every chatbot to behave: tone, length, "
        "level of jargon, format.",
    )
    bot_instructions: str = Field(
        "",
        description="How the learner wants this particular chatbot to behave, "
        "including topics or kinds of results they asked not to be shown.",
    )


EXTRACTION_INSTRUCTIONS = """You maintain notes about a learner who uses MIT Open
Learning chatbots. You are given the chatbot that was in use, the current notes, the
learner's latest message and the chatbot's reply. Return the revised notes.

Rules:
- Record only the learner's own statements. The assistant's reply is context for
  understanding the learner's message and never a source of facts or preferences.
- Only durable information: who the learner is, what they want in general, how they
  want to be helped. Skip anything that only matters for this one thread, such as the
  specific course, lecture or question at hand.
- 'about' is for facts about the person. 'instructions' is for how every chatbot should
  behave. 'bot_instructions' is for preferences that only make sense for the chatbot in
  use, including things they asked not to be shown.
- Never record names, emails, problem statements, attempted or correct answers, hints,
  grades, scores, or problem identifiers.
- Keep each section under {max_chars} characters. Drop anything the learner contradicts.
  If the message adds nothing durable, return the notes unchanged."""


def memory_namespace(global_id: str) -> tuple[str, str]:
    return ("memories", global_id)


def bot_instructions_key(bot_name: str) -> str:
    return f"{INSTRUCTIONS_KEY}:{bot_name}"


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


def load_learner_memory(global_id: str, bot_name: str) -> LearnerMemory:
    store = DjangoMemoryStore()
    ns = memory_namespace(global_id)

    def text(key):
        item = store.get(ns, key)
        return item.value.get("text", "") if item else ""

    return LearnerMemory(
        about=text(ABOUT_KEY),
        instructions=text(INSTRUCTIONS_KEY),
        bot_instructions=text(bot_instructions_key(bot_name)),
    )


def save_learner_memory(global_id: str, bot_name: str, mem: LearnerMemory) -> None:
    """Write the sections that changed; never create empty ones."""
    store = DjangoMemoryStore()
    ns = memory_namespace(global_id)
    current = load_learner_memory(global_id, bot_name)
    for key, revised, old in (
        (ABOUT_KEY, mem.about, current.about),
        (INSTRUCTIONS_KEY, mem.instructions, current.instructions),
        (
            bot_instructions_key(bot_name),
            mem.bot_instructions,
            current.bot_instructions,
        ),
    ):
        text = revised.strip()[: settings.AI_MEMORY_MAX_CHARS]
        if text and text != old:
            store.put(ns, key, {"text": text})


def _profile_lines(bot_name: str, profile: dict) -> list[str]:
    rows = [("Education", profile.get("current_education"))]
    if bot_name in FULL_PROFILE_BOTS:
        rows += [
            (
                "Topics of interest",
                [t["name"] for t in profile.get("topic_interests", [])],
            ),
            ("Goals", profile.get("goals")),
            ("Certificate", CERTIFICATE_TEXT.get(profile.get("certificate_desired"))),
            ("Time commitment", profile.get("time_commitment")),
            ("Preferred delivery", profile.get("delivery")),
        ]
    return [
        f"- {label}: {', '.join(v) if isinstance(v, list) else v}"
        for label, v in rows
        if v
    ]


def build_learner_context(bot_name: str, profile: dict, mem: LearnerMemory) -> str:
    """Render the block for one bot; '' when nothing is known."""
    cap = settings.AI_MEMORY_MAX_CHARS
    about = _profile_lines(bot_name, profile)
    if mem.about:
        about.append(mem.about[:cap])
    instructions = [t[:cap] for t in (mem.instructions, mem.bot_instructions) if t]
    if not about and not instructions:
        return ""
    lines = [CONTEXT_INSTRUCTION]
    if about:
        lines += [ABOUT_HEADING, *about]
    if instructions:
        lines += [INSTRUCTIONS_HEADING, *instructions]
    return "\n".join(lines)


def get_learner_context(user, bot_name: str) -> str:
    """Return the learner context block, or '' if disabled or anonymous."""
    if not memory_enabled(user):
        return ""
    try:
        return build_learner_context(
            bot_name,
            fetch_learner_profile(user.global_id),
            load_learner_memory(user.global_id, bot_name),
        )
    except Exception:
        log.exception("Learner context unavailable for %s", user.global_id)
        return ""


def extract_learner_memory(
    global_id: str, bot_name: str, message: str, response: str
) -> None:
    """One structured LLM call revises the three sections from the latest exchange."""
    current = load_learner_memory(global_id, bot_name)
    llm = init_chat_model(settings.AI_MEMORY_EXTRACTION_MODEL, temperature=0)
    revised = llm.with_structured_output(LearnerMemory).invoke(
        [
            SystemMessage(
                EXTRACTION_INSTRUCTIONS.format(max_chars=settings.AI_MEMORY_MAX_CHARS)
            ),
            HumanMessage(
                f"Chatbot in use: {bot_name}\n\n"
                f"Current notes:\n{current.model_dump_json(indent=1)}\n\n"
                f"Learner's message:\n{message}\n\n"
                f"Chatbot's reply (context only):\n{response[:2000]}"
            ),
        ]
    )
    save_learner_memory(global_id, bot_name, revised)
