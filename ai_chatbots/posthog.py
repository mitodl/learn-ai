"""PostHog serialization and callback handler for AI chatbots."""

import logging
from typing import Any, Optional
from uuid import UUID

import litellm
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from posthog.ai.langchain import CallbackHandler

from ai_chatbots.api import serialize_tool_calls

log = logging.getLogger(__name__)


def serialize_for_posthog(obj: Any) -> Any:  # noqa: PLR0911
    """
    Recursively serialize objects to JSON-compatible format for PostHog.
    """
    # Handle primitive types first
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj

    # Handle BaseMessage objects from LangChain
    if hasattr(obj, "type") and hasattr(obj, "content"):
        msg_dict = {"role": obj.type, "type": obj.type, "content": obj.content}
        if hasattr(obj, "id"):
            msg_dict["id"] = obj.id
        if hasattr(obj, "additional_kwargs"):
            msg_dict["additional_kwargs"] = serialize_for_posthog(obj.additional_kwargs)
        if hasattr(obj, "tool_calls"):
            msg_dict["tool_calls"] = serialize_tool_calls(obj.tool_calls)
        return msg_dict

    # Handle Send objects from LangGraph
    if type(obj).__name__ == "Send":
        return {
            "node": obj.node if hasattr(obj, "node") else str(obj),
            "arg": serialize_for_posthog(obj.arg if hasattr(obj, "arg") else {}),
        }

    # Handle collections
    if isinstance(obj, list):
        return [serialize_for_posthog(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize_for_posthog(v) for k, v in obj.items()}

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: serialize_for_posthog(v) for k, v in obj.__dict__.items()}

    # Fallback to string representation
    return str(obj)


def format_posthog_messages(messages: list[BaseMessage]) -> list[dict]:
    """
    Standardize Langchain Message objects to a list of dicts.
    """
    flattened_messages = []
    for message_list in messages:
        flattened_messages.extend(message_list)
    return serialize_for_posthog(flattened_messages)


class TokenTrackingCallbackHandler(CallbackHandler):
    """
    PostHog callback handler that tracks token counts for cost
    calculation using LiteLLM's token_counter
    """

    def __init__(self, model_name: str, **kwargs):
        self.bot = kwargs.pop("bot", None)
        super().__init__(**kwargs)
        self.model_name = model_name
        self.input_tokens = 0
        self.set_trace_attributes()

    def set_trace_attributes(self):
        """Set trace attributes for PostHog"""
        self._ph_client.capture(
            event="$ai_trace",
            distinct_id=self.bot.user_id,
            properties={
                "$ai_trace_id": self.bot.thread_id,
                "$ai_span_name": self.bot.JOB_ID,
                "botName": self.bot.JOB_ID,
            },
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ):
        """Format messages and estimate input tokens"""
        posthog_messages = format_posthog_messages(messages)
        try:
            # Use LiteLLM token_counter with the proper format
            self.input_tokens = litellm.token_counter(
                model=self.model_name, messages=posthog_messages
            )
        except Exception:
            # Fallback to character-based estimation
            total_input_chars = 0
            for message_list in messages:
                for message in message_list:
                    if hasattr(message, "content"):
                        total_input_chars += len(str(message.content))
                    else:
                        total_input_chars += len(str(message))
            self.input_tokens = total_input_chars // 4
            log.exception("LiteLLM token_counter failed, using character estimation")
        if not hasattr(self, "_properties") or self._properties is None:
            self._properties = {}
        self._properties.update(
            {
                "question": (
                    [
                        msg["content"]
                        for msg in posthog_messages
                        if msg["type"] == "human"
                    ]
                    or [""]
                )[-1],
                "$ai_input": posthog_messages,
            }
        )

        # Call parent method
        super().on_chat_model_start(
            serialized, messages, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ):
        # Calculate output tokens using LiteLLM's token_counter
        output_tokens = 0

        try:
            # Collect all output text
            output_text = (
                "".join(
                    [
                        gen_chunk.text
                        for generation in response.generations
                        for gen_chunk in generation
                        if hasattr(gen_chunk, "text") and gen_chunk.text
                    ]
                )
                if hasattr(response, "generations") and response.generations
                else ""
            )

            # Use LiteLLM token_counter for the complete output text
            if output_text:
                output_tokens = litellm.token_counter(
                    model=self.model_name, text=output_text
                )
        except Exception:
            # Fallback to character-based estimation
            if hasattr(response, "generations") and response.generations:
                for generation in response.generations:
                    for gen_chunk in generation:
                        if hasattr(gen_chunk, "text") and gen_chunk.text:
                            output_tokens += len(gen_chunk.text) // 4
            log.exception(
                "token_counter failed, using character estimation for tokens."
            )
        self._properties.update(
            {
                "answer": output_text,
                "$ai_input_tokens": self.input_tokens,
                "$ai_output_tokens": output_tokens,
                "$ai_trace_name": self.bot.JOB_ID,
                "$ai_span_name": self.bot.JOB_ID,
            }
        )
        # Call parent method
        super().on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def _pop_run_and_capture_trace_or_span(
        self, run_id: UUID, parent_run_id: Optional[UUID], outputs: Any
    ):
        """Override to serialize outputs before passing to parent."""
        serialized_outputs = serialize_for_posthog(outputs)
        super()._pop_run_and_capture_trace_or_span(
            run_id, parent_run_id, serialized_outputs
        )
