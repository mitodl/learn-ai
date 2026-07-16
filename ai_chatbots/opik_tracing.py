"""Cost-aware Opik LangChain tracer.

Opik's LangChain integration cannot price our LLM spans on its own. Its provider
usage extractors gate on provider-specific serialized kwargs (e.g. the OpenAI
extractor requires an ``openai_api_key`` entry in ``run.serialized.kwargs``), but
``ChatLiteLLM`` serializes as ``not_implemented`` with no kwargs. No extractor
matches, so Opik logs the span with *no* usage, model, provider, or cost -- even
though the token usage is present in the LangChain run.

``CostTrackingOpikTracer`` fills that gap. After Opik logs the span, it reads the
token usage and model name straight from the run, derives the provider and bare
model id, and re-sends the *same* span (via the same internal API Opik itself
uses) with ``usage`` / ``provider`` / ``model`` attached. The span keeps its
name, type, input and output, and Opik's backend computes the estimated cost
from the (provider, model, usage) triple.

The model name arrives proxy-prefixed (``litellm_proxy/<provider>/<model>``,
e.g. ``litellm_proxy/openai/gpt-4o-mini``); we split it into the provider
(``openai``) and the bare model id (``gpt-4o-mini``) that Opik's price map uses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opik import tracing_runtime_config
from opik.integrations.langchain import OpikTracer

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run

log = logging.getLogger(__name__)

_PROXY_PREFIX = "litellm_proxy/"


def parse_provider_and_model(
    ls_model_name: str | None,
) -> tuple[str | None, str | None]:
    """Split a proxy-routed model id into (provider, bare model) for pricing.

    ``litellm_proxy/openai/gpt-4o-mini``   -> ("openai", "gpt-4o-mini")
    ``litellm_proxy/anthropic/claude-3-5`` -> ("anthropic", "claude-3-5")
    ``openai/gpt-4o-mini``                 -> ("openai", "gpt-4o-mini")
    ``gpt-4o-mini``                        -> (None, "gpt-4o-mini")
    """
    if not ls_model_name:
        return None, None
    name = ls_model_name
    if name.startswith(_PROXY_PREFIX):
        name = name[len(_PROXY_PREFIX) :]
    if "/" in name:
        provider, model = name.split("/", 1)
        return provider or None, model or None
    return None, name


def _find_first(obj: Any, key: str) -> Any:
    """Return the first non-empty value stored under ``key`` in a nested run."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and v:
                return v
            found = _find_first(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first(item, key)
            if found is not None:
                return found
    return None


def _to_openai_usage(usage_metadata: dict[str, Any]) -> dict[str, Any]:
    """Map LangChain ``usage_metadata`` to the OpenAI keys Opik expects."""
    return {
        "prompt_tokens": usage_metadata.get("input_tokens"),
        "completion_tokens": usage_metadata.get("output_tokens"),
        "total_tokens": usage_metadata.get("total_tokens"),
    }


def _extract_llm_span_info(run: Run) -> dict[str, Any] | None:
    """Pull usage + provider + model out of a LangChain LLM run, or None."""
    if getattr(run, "run_type", None) not in ("llm", "chat_model"):
        return None
    run_dict = run.dict()
    usage_metadata = _find_first(run_dict.get("outputs"), "usage_metadata")
    ls_model_name = (
        (run_dict.get("extra") or {}).get("metadata", {}).get("ls_model_name")
    )
    if not usage_metadata and not ls_model_name:
        return None
    provider, model = parse_provider_and_model(ls_model_name)
    return {
        "usage": _to_openai_usage(usage_metadata) if usage_metadata else None,
        "provider": provider,
        "model": model,
    }


class CostTrackingOpikTracer(OpikTracer):
    """OpikTracer that attaches usage/provider/model so Opik can compute cost.

    Opik's own extraction does not recognize ``ChatLiteLLM`` runs, so we let it
    log the span normally, then re-send the same span with the cost fields we
    read from the run. We reuse Opik's ``__internal_api__span__`` upsert -- the
    same call the base tracer makes -- so the span retains its name, type, and
    input/output while gaining usage, provider, and model.
    """

    def _process_end_span(self, run: Run) -> None:
        info = _extract_llm_span_info(run)
        span_data = self._span_data_map.get(run.id)
        super()._process_end_span(run)
        self._attach_cost_fields(span_data, info)

    def _attach_cost_fields(self, span_data: Any, info: dict[str, Any] | None) -> None:
        if not info or span_data is None:
            return
        if info["usage"] is None and info["model"] is None:
            return
        try:
            update_kwargs = {
                k: v
                for k, v in {
                    "usage": info.get("usage"),
                    "model": info.get("model"),
                    "provider": info.get("provider"),
                }.items()
                if v is not None
            }
            if not update_kwargs:
                return
            span_data.update(**update_kwargs)
            if tracing_runtime_config.is_tracing_active():
                self._opik_client.__internal_api__span__(**span_data.as_parameters)
        except Exception:
            log.exception("Failed to attach Opik cost fields to span")
