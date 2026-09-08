# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import MagicMock

import agents.tracing
import pytest
from agents.tracing import agent_span, guardrail_span, trace
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

from opentelemetry.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    ToolInvocation,
)


class _Span:
    """Minimal stand-in for agents-library Span (must be weakref-able)."""

    def __init__(self, span_data: Any) -> None:
        self.span_data = span_data
        # Span objects in tests don't need a span_id since the processor
        # keys its WeakKeyDictionary by the Span instance itself, but
        # keep one around for parity with the real class.
        self.span_id = f"span-{id(self)}"


class _Trace:
    """Minimal stand-in for agents-library Trace."""

    def __init__(self, trace_id: str, name: str) -> None:
        self.trace_id = trace_id
        self.name = name


def _build_handler() -> MagicMock:
    return MagicMock()


def test_trace_start_end_creates_and_stops_workflow() -> None:
    handler = _build_handler()
    handler.workflow.return_value = MagicMock(attributes={})
    processor = GenAITracingProcessor(handler, provider="openai")
    trace = _Trace("trace-1", "Agent workflow")

    processor.on_trace_start(trace)
    handler.workflow.assert_called_once_with(name="Agent workflow")
    workflow_invocation = handler.workflow.return_value
    assert (
        workflow_invocation.attributes["gen_ai.workflow.name"]
        == "Agent workflow"
    )

    processor.on_trace_end(trace)
    workflow_invocation.stop.assert_called_once_with()


def test_agent_span_creates_invoke_local_agent() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(AgentSpanData(name="triage"))

    processor.on_span_start(span)
    handler.invoke_local_agent.assert_called_once_with(agent_name="triage")

    processor.on_span_end(span)
    handler.invoke_local_agent.return_value.stop.assert_called_once_with()


@pytest.mark.parametrize(
    ("triggered", "verdict"),
    [(False, "allow"), (True, "deny")],
)
def test_guardrail_span_emits_result_event(
    triggered: bool, verdict: str
) -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )
    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(GuardrailSpanData(name="content_filter", triggered=False))

    with tracer_provider.get_tracer(__name__).start_as_current_span(
        "enclosing"
    ) as enclosing_span:
        processor.on_span_start(span)
        span.span_data.triggered = triggered
        processor.on_span_end(span)

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.name == "enclosing"

    (readable_log,) = log_exporter.get_finished_logs()
    log_record = readable_log.log_record
    assert log_record.attributes == {
        "gen_ai.guardrail.component.name": "content_filter",
        "gen_ai.provider.name": "openai",
        "gen_ai.guardrail.verdict.type": verdict,
    }
    assert log_record.trace_id == enclosing_span.get_span_context().trace_id
    assert log_record.span_id == enclosing_span.get_span_context().span_id


def test_guardrail_span_error_emits_result_event_without_verdict() -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )
    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(GuardrailSpanData(name="content_filter", triggered=False))
    span.error = {"message": "Error running guardrail", "data": {}}

    with tracer_provider.get_tracer(__name__).start_as_current_span(
        "enclosing"
    ):
        processor.on_span_start(span)
        processor.on_span_end(span)

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.name == "enclosing"

    (readable_log,) = log_exporter.get_finished_logs()
    log_record = readable_log.log_record
    assert log_record.attributes == {
        "gen_ai.guardrail.component.name": "content_filter",
        "gen_ai.provider.name": "openai",
        "error.type": "_OTHER",
    }


def test_guardrail_event_uses_real_agent_span_as_parent() -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )
    processor = GenAITracingProcessor(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        ),
        provider="openai",
    )
    trace_provider = agents.tracing.get_trace_provider()
    multi = getattr(trace_provider, "_multi_processor", None)
    previous_processors = list(getattr(multi, "_processors", ()))
    try:
        agents.tracing.set_trace_processors([processor])
        with trace("workflow"):
            with agent_span("triage"):
                with guardrail_span("content_filter"):
                    pass
    finally:
        agents.tracing.set_trace_processors(previous_processors)

    agent_record = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get("gen_ai.operation.name") == "invoke_agent"
    )
    (readable_log,) = log_exporter.get_finished_logs()
    assert readable_log.log_record.span_id == agent_record.context.span_id


def test_function_span_creates_tool_invocation_and_sets_provider_metric() -> (
    None
):
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation,
        metric_attributes={},
        should_capture_content_on_span=True,
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(
        FunctionSpanData(
            name="get_weather",
            input=None,
            output=None,
        )
    )

    processor.on_span_start(span)
    handler.tool.assert_called_once_with(
        name="get_weather",
        tool_type="function",
    )
    tool_invocation = handler.tool.return_value

    assert (
        tool_invocation.metric_attributes["gen_ai.provider.name"] == "openai"
    )

    # Input and output both get populated on the agents library span_data
    # while the tool runs, i.e. after on_span_start; our on_span_end reads
    # them.
    span.span_data.input = '{"city":"BCN"}'
    span.span_data.output = "sunny"
    processor.on_span_end(span)
    assert tool_invocation.arguments == {"city": "BCN"}
    assert tool_invocation.tool_result == "sunny"
    tool_invocation.stop.assert_called_once_with()


def test_function_span_skips_content_when_capture_disabled() -> None:
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation,
        metric_attributes={},
        should_capture_content_on_span=False,
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="get_weather", input=None, output=None))

    processor.on_span_start(span)
    span.span_data.input = '{"city":"BCN"}'
    span.span_data.output = "sunny"
    processor.on_span_end(span)

    tool_invocation = handler.tool.return_value
    # A spec'd mock rejects reads of attributes nothing assigned, so these
    # assert the processor skipped the serialization work entirely.
    with pytest.raises(AttributeError):
        _ = tool_invocation.arguments
    with pytest.raises(AttributeError):
        _ = tool_invocation.tool_result
    tool_invocation.stop.assert_called_once_with()


def test_function_span_without_output_still_stops() -> None:
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation,
        metric_attributes={},
        should_capture_content_on_span=True,
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="noop", input=None, output=None))

    processor.on_span_start(span)
    processor.on_span_end(span)
    tool_invocation = handler.tool.return_value
    # tool_result stays as whatever MagicMock default; what matters is
    # we didn't crash and we stopped.
    tool_invocation.stop.assert_called_once_with()


def test_generation_and_response_spans_ignored() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")

    for span_data in (
        GenerationSpanData(model="gpt-4o-mini"),
        ResponseSpanData(),
    ):
        span = _Span(span_data)
        processor.on_span_start(span)
        processor.on_span_end(span)

    handler.invoke_local_agent.assert_not_called()
    handler.tool.assert_not_called()
    handler.inference.assert_not_called()


def test_handoff_emits_raw_span() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(
        HandoffSpanData(from_agent="triage", to_agent="weather_specialist")
    )
    # Doesn't raise; the actual OTel span emission is verified end-to-end
    # by the conformance scenario.
    processor.on_span_start(span)
    processor.on_span_end(span)


def test_shutdown_stops_open_invocations() -> None:
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation, metric_attributes={}
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    # Hold strong references to the trace / span objects so the
    # WeakKeyDictionary entries survive until shutdown runs.
    trace = _Trace("t", "wf")
    agent_span = _Span(AgentSpanData(name="agent"))
    tool_span = _Span(
        FunctionSpanData(name="get_weather", input=None, output=None)
    )
    processor.on_trace_start(trace)
    processor.on_span_start(agent_span)
    processor.on_span_start(tool_span)
    assert len(processor._invocations) == 3

    processor.shutdown()

    handler.workflow.return_value.stop.assert_called_once_with()
    handler.invoke_local_agent.return_value.stop.assert_called_once_with()
    handler.tool.return_value.stop.assert_called_once_with()
    assert len(processor._invocations) == 0


def test_tool_content_captured_from_span_end(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arguments the agents library fills in mid-span still land on the span."""
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="get_weather", input=None, output=None))

    processor.on_span_start(span)
    span.span_data.input = '{"city":"Barcelona"}'
    span.span_data.output = "sunny"
    processor.on_span_end(span)

    (tool_span,) = span_exporter.get_finished_spans()
    assert tool_span.attributes is not None
    assert (
        tool_span.attributes["gen_ai.tool.call.arguments"]
        == '{"city":"Barcelona"}'
    )
    assert tool_span.attributes["gen_ai.tool.call.result"] == "sunny"


@pytest.mark.parametrize(
    ("span_data_input", "expected"),
    [
        # The spacing LiteLLM produces for Anthropic normalizes to the same
        # value as the compact JSON the OpenAI APIs forward.
        ('{"city": "Barcelona"}', '{"city":"Barcelona"}'),
        ('{"city":"Barcelona"}', '{"city":"Barcelona"}'),
        # A provider may emit something that isn't valid JSON, or valid JSON
        # that isn't an object; neither may change the attribute's type.
        ("city=Barcelona", "city=Barcelona"),
        ("[1,2]", "[1,2]"),
        ("42", "42"),
        ("null", "null"),
        # No arguments to record: `trace_include_sensitive_data=False` leaves
        # `input` None, and tool types without arguments leave it empty.
        (None, None),
        ("", None),
    ],
)
def test_tool_arguments_value_shapes(
    span_data_input: str | None,
    expected: str | None,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="get_weather", input=None, output=None))

    processor.on_span_start(span)
    span.span_data.input = span_data_input
    processor.on_span_end(span)

    (tool_span,) = span_exporter.get_finished_spans()
    assert tool_span.attributes is not None
    if expected is None:
        assert "gen_ai.tool.call.arguments" not in tool_span.attributes
    else:
        assert tool_span.attributes["gen_ai.tool.call.arguments"] == expected


def test_no_content_captured_when_capture_env_unset(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, raising=False
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="get_weather", input=None, output=None))

    processor.on_span_start(span)
    span.span_data.input = '{"city":"Barcelona"}'
    span.span_data.output = "sunny"
    processor.on_span_end(span)

    (tool_span,) = span_exporter.get_finished_spans()
    assert tool_span.attributes is not None
    assert "gen_ai.tool.call.arguments" not in tool_span.attributes
    assert "gen_ai.tool.call.result" not in tool_span.attributes
    # The non-content tool attributes are still present.
    assert tool_span.attributes["gen_ai.tool.name"] == "get_weather"


def test_tool_span_error_sets_error_status_and_type(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    """A tool the agents library recorded as failed must not look successful.

    The agents library reports tool failures on ``Span.error``. If the
    processor ignores it the exported span is byte-identical to a
    successful one, and a backend cannot filter or count failed tool
    calls.
    """
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="get_weather", input=None, output=None))
    span.error = {"message": "Error running tool", "data": {}}

    processor.on_span_start(span)
    processor.on_span_end(span)

    (tool_span,) = span_exporter.get_finished_spans()
    assert tool_span.status.status_code is StatusCode.ERROR
    assert tool_span.attributes is not None
    assert tool_span.attributes["error.type"] == "_OTHER"


def test_agent_span_error_sets_error_status_and_type(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    """A fatal tool also ends the surrounding agent span with an error."""
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(AgentSpanData(name="weather_agent"))
    span.error = {"message": "Error in agent run", "data": {}}

    processor.on_span_start(span)
    processor.on_span_end(span)

    (agent_span,) = span_exporter.get_finished_spans()
    assert agent_span.status.status_code is StatusCode.ERROR
    assert agent_span.attributes is not None
    assert agent_span.attributes["error.type"] == "_OTHER"


def test_state_uses_weakref_so_dropped_spans_are_collected() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span: _Span | None = _Span(AgentSpanData(name="triage"))
    processor.on_span_start(span)
    assert len(processor._invocations) == 1

    span = None  # drop the only strong reference
    gc.collect()

    assert len(processor._invocations) == 0
