"""Message truncation for managing long chat sessions."""

import logging
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.utils.runnable import RunnableCallable

log = logging.getLogger(__name__)


class MessageTruncationNode(RunnableCallable):
    """
    Truncates message history to keep only the last N human messages and
    their responses.

    Preserves all messages in the database, but only sends recent messages
    to the LLM to prevent context window errors. Counts only human messages
    to determine truncation point, then includes all associated AI and tool
    messages.
    """

    def __init__(
        self,
        *,
        max_human_messages: int = 10,
        input_messages_key: str = "messages",
        output_messages_key: str = "llm_input_messages",
        name: str = "truncation",
    ) -> None:
        """
        Initialize the MessageTruncationNode.

        Args:
            max_human_messages: Maximum number of human messages to keep.
            input_messages_key: Key to read messages from state.
            output_messages_key: Key to write truncated messages to state.
            name: Name of the node.
        """
        super().__init__(self._func, name=name, trace=False)
        self.max_human_messages = max_human_messages
        self.input_messages_key = input_messages_key
        self.output_messages_key = output_messages_key

    def _func(self, node_input: dict[str, Any]) -> dict[str, Any]:
        """Truncate messages to last N human messages and their responses."""
        messages = node_input.get(self.input_messages_key, [])

        if not messages:
            return {self.output_messages_key: messages}

        # Separate system message from others
        system_msg = None
        other_messages = messages

        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            other_messages = messages[1:]

        # Count human messages and find the truncation point
        human_message_count = sum(
            1 for msg in other_messages if isinstance(msg, HumanMessage)
        )

        if human_message_count <= self.max_human_messages:
            # No truncation needed
            return {self.output_messages_key: messages}

        # Find the index of the Nth-from-last human message
        truncation_index = self.find_nth_human_message_from_end(
            other_messages, self.max_human_messages
        )

        # Truncate from that point onward
        truncated = other_messages[truncation_index:]

        # Reassemble with system message
        result = [system_msg, *truncated] if system_msg else truncated

        log.info(
            "Truncated messages: %d -> %d (kept last %d human messages)",
            len(messages),
            len(result),
            self.max_human_messages,
        )

        return {self.output_messages_key: result}

    def find_nth_human_message_from_end(
        self, messages: list[BaseMessage], n: int
    ) -> int:
        """
        Find the index of the Nth human message from the end.

        Args:
            messages: List of messages to search.
            n: Number of human messages to count from the end.

        Returns:
            Index of the Nth human message from the end.
        """
        human_indices = [
            i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)
        ]

        if len(human_indices) < n:
            # Not enough human messages, return start
            return 0

        # Return the index of the Nth-from-last human message
        return human_indices[-n]
