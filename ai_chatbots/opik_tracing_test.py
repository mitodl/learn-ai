"""Tests for the cost-aware Opik LangChain tracer."""

import pytest
from opik.integrations.langchain import OpikTracer

from ai_chatbots import opik_tracing
from ai_chatbots.opik_tracing import (
    CostTrackingOpikTracer,
    _extract_llm_span_info,
    parse_provider_and_model,
)

DEFAULT_USAGE = {"input_tokens": 14, "output_tokens": 6, "total_tokens": 20}
OPENAI_USAGE = {"prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20}


class FakeRun:
    """Stand-in for a LangChain Run: exposes attributes and a ``dict()``."""

    def __init__(self, run_id, run_type, extra, outputs):
        """Store the run fields Opik reads."""
        self.id = run_id
        self.run_type = run_type
        self.extra = extra
        self.outputs = outputs

    def dict(self):
        """Return the run as the nested dict Opik expects from ``run.dict()``."""
        return {
            "run_type": self.run_type,
            "extra": self.extra,
            "outputs": self.outputs,
        }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("litellm_proxy/openai/gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("litellm_proxy/anthropic/claude-3-5", ("anthropic", "claude-3-5")),
        ("openai/gpt-4o-mini", ("openai", "gpt-4o-mini")),
        ("bedrock/anthropic.claude-v2", ("bedrock", "anthropic.claude-v2")),
        ("gpt-4o-mini", (None, "gpt-4o-mini")),
        ("", (None, None)),
        (None, (None, None)),
    ],
)
def test_parse_provider_and_model(raw, expected):
    """The proxy prefix is stripped and provider/model are split for pricing."""
    assert parse_provider_and_model(raw) == expected


def _make_run(model="litellm_proxy/openai/gpt-4o-mini", run_type="llm", usage=None):
    """Build a run resembling LangChain's serialized ChatLiteLLM run output."""
    return FakeRun(
        "run-1",
        run_type,
        {"metadata": {"ls_model_name": model, "ls_provider": "litellm"}},
        {
            "generations": [
                [
                    {
                        "text": "hi",
                        "message": {
                            "kwargs": {"usage_metadata": usage or DEFAULT_USAGE}
                        },
                    }
                ]
            ]
        },
    )


def test_extract_llm_span_info_maps_usage_provider_model():
    """Usage is mapped to OpenAI keys; provider/model derived from the name."""
    info = _extract_llm_span_info(_make_run())
    assert info == {"usage": OPENAI_USAGE, "provider": "openai", "model": "gpt-4o-mini"}


def test_extract_llm_span_info_finds_deeply_nested_usage():
    """usage_metadata is located wherever it sits in the run tree."""
    run = _make_run()
    # Bury usage one level deeper to exercise the recursive search.
    run.outputs = {"deeper": {"generations": [run.outputs]}}
    info = _extract_llm_span_info(run)
    assert info["usage"]["total_tokens"] == 20


def test_extract_llm_span_info_skips_non_llm_runs():
    """Non-LLM runs (chains/tools) are ignored."""
    assert _extract_llm_span_info(_make_run(run_type="chain")) is None


def test_extract_llm_span_info_none_without_usage_or_model():
    """A run with neither usage nor a model name yields nothing to attach."""
    run = FakeRun("x", "llm", {}, {})
    assert _extract_llm_span_info(run) is None


def _make_tracer(mocker, client):
    """Instantiate the tracer without running OpikTracer.__init__."""
    mocker.patch.object(
        OpikTracer,
        "_opik_client",
        new_callable=mocker.PropertyMock,
        return_value=client,
    )
    return object.__new__(CostTrackingOpikTracer)


def test_tracer_attaches_cost_fields_after_super(mocker):
    """The tracer logs via super, then re-sends the span with cost fields."""
    super_end = mocker.patch.object(OpikTracer, "_process_end_span")
    mocker.patch.object(
        opik_tracing.tracing_runtime_config, "is_tracing_active", return_value=True
    )
    client = mocker.Mock()
    client.__internal_api__span__ = mocker.Mock()
    run = _make_run()
    span_data = mocker.Mock()
    span_data.as_parameters = {"id": "span-1"}

    tracer = _make_tracer(mocker, client)
    tracer._span_data_map = {run.id: span_data}  # noqa: SLF001

    tracer._process_end_span(run)  # noqa: SLF001

    super_end.assert_called_once_with(run)
    span_data.update.assert_called_once_with(
        usage=OPENAI_USAGE, model="gpt-4o-mini", provider="openai"
    )
    client.__internal_api__span__.assert_called_once_with(id="span-1")


def test_tracer_noop_when_no_llm_info(mocker):
    """Non-LLM runs still delegate to super but attach nothing."""
    super_end = mocker.patch.object(OpikTracer, "_process_end_span")
    client = mocker.Mock()
    client.__internal_api__span__ = mocker.Mock()
    run = _make_run(run_type="chain")
    span_data = mocker.Mock()

    tracer = _make_tracer(mocker, client)
    tracer._span_data_map = {run.id: span_data}  # noqa: SLF001

    tracer._process_end_span(run)  # noqa: SLF001

    super_end.assert_called_once_with(run)
    span_data.update.assert_not_called()
    client.__internal_api__span__.assert_not_called()
