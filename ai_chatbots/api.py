"""AI-specific functions for ai_agents."""

import json
import logging

from django.conf import settings
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from typing_extensions import TypedDict

from main.utils import Singleton

log = logging.getLogger(__name__)


class ChatMemory(metaclass=Singleton):
    """
    Singleton class to manage chat memory

    For now, MemorySaver will be the default for temporary memory retention.
    Chat history will be lost on server restarts.

    A RedisSaver could be added later to make it somewhat less temporary:
    https://langchain-ai.github.io/langgraph/how-tos/persistence_redis/

    For production, persistent memory should be used. PostgresSaver can be
    used for now as a test but we should create a Django ORM-friendly alternate
    implementation before it is truly live.
    """

    def __init__(self):
        self.checkpointer = (
            get_postgres_saver() if settings.AI_PERSISTENT_MEMORY else MemorySaver()
        )


def persistence_db(db_name: str | None = None) -> str:
    """
    Return the database connection string needed by langgraph PostgresSaver.
    """
    db = settings.DATABASES[db_name or "default"]
    sslmode = "disable" if db["DISABLE_SERVER_SIDE_CURSORS"] else "require"
    return f"postgresql://{db['USER']}:{db['PASSWORD']}@{db['HOST']}:{db['PORT']}/{db['NAME']}?sslmode={sslmode}"


def get_postgres_saver() -> AsyncPostgresSaver:
    """
    Return a PostgresSaver instance for persistent chat memory.
    For local testing purposes only, we should create our
    own version that uses the Django ORM.
    """
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    pool = AsyncConnectionPool(
        conninfo=persistence_db(),
        max_size=settings.AI_PERSISTENT_POOL_SIZE,
        kwargs=connection_kwargs,
    )
    return AsyncPostgresSaver(pool)


def get_search_tool_metadata(thread_id: str, latest_state: TypedDict) -> str:
    """
    Return the metadata for a bot search tool.
    """
    tool_messages = (
        []
        if not latest_state
        else [
            t
            for t in latest_state.values.get("messages", [])
            if t and t.__class__ == ToolMessage
        ]
    )
    if tool_messages:
        msg_content = tool_messages[-1].content
        try:
            content = json.loads(msg_content or {})
            metadata = {
                "metadata": {
                    "search_parameters": content.get("metadata", {}).get(
                        "parameters", []
                    ),
                    "search_results": content.get("results", []),
                    "thread_id": thread_id,
                }
            }
            return json.dumps(metadata)
        except json.JSONDecodeError:
            log.exception("Error parsing tool metadata, not valid JSON")
            return json.dumps(
                {"error": "Error parsing tool metadata", "content": msg_content}
            )
    else:
        return "{}"
