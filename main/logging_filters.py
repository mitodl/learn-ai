"""Logging filters for third-party log noise."""

import logging


class DropFailedToDetachContext(logging.Filter):
    """
    Drop opentelemetry.context "Failed to detach context" error records.

    openlit's LangChain callback handler attaches an OTel context token in
    on_llm_start and detaches it in on_llm_end/on_llm_error. Under async
    execution LangChain runs each callback in its own asyncio task (or
    executor thread), so the detach raises ValueError("... was created in a
    different Context"), which OpenTelemetry catches and logs with a full
    traceback on every LLM call. Spans are unaffected (openlit ends them
    explicitly by run_id), so the record is pure noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for the detach-failure record, True otherwise."""
        return not record.getMessage().startswith("Failed to detach context")
