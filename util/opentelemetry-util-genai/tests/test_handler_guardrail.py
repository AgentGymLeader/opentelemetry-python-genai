# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

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
from opentelemetry.util.genai._guardrail_event import (
    _GUARDRAIL_RESULT_EVENT_NAME,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler


@pytest.mark.parametrize(
    ("triggered", "verdict"),
    [(True, "deny"), (False, "allow")],
)
def test_guardrail_emits_result_event_on_current_span(
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

    with tracer_provider.get_tracer(__name__).start_as_current_span(
        "enclosing"
    ) as enclosing_span:
        handler.guardrail_result(
            "content_filter", "openai", triggered=triggered
        )

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.name == "enclosing"

    (readable_log,) = log_exporter.get_finished_logs()
    log_record = readable_log.log_record
    assert log_record.event_name == _GUARDRAIL_RESULT_EVENT_NAME
    assert log_record.attributes == {
        "gen_ai.guardrail.component.name": "content_filter",
        "gen_ai.provider.name": "openai",
        "gen_ai.guardrail.verdict.type": verdict,
    }
    assert log_record.trace_id == enclosing_span.get_span_context().trace_id
    assert log_record.span_id == enclosing_span.get_span_context().span_id


def test_guardrail_error_emits_result_event_without_verdict() -> None:
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

    with tracer_provider.get_tracer(__name__).start_as_current_span(
        "enclosing"
    ):
        handler.guardrail_result(
            "content_filter",
            "openai",
            triggered=False,
            error_type="_OTHER",
        )

    (finished_span,) = span_exporter.get_finished_spans()
    assert finished_span.name == "enclosing"

    (readable_log,) = log_exporter.get_finished_logs()
    log_record = readable_log.log_record
    assert log_record.event_name == _GUARDRAIL_RESULT_EVENT_NAME
    assert log_record.attributes == {
        "gen_ai.guardrail.component.name": "content_filter",
        "gen_ai.provider.name": "openai",
        "error.type": "_OTHER",
    }


@pytest.mark.parametrize(
    ("emit_event", "expected_count"),
    [(None, 1), ("true", 1), ("TrUe", 1), ("false", 0), ("FaLsE", 0)],
)
def test_guardrail_result_event_gate(
    emit_event: str | None,
    expected_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if emit_event is None:
        monkeypatch.delenv(
            OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, raising=False
        )
    else:
        monkeypatch.setenv(OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT, emit_event)
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )
    handler = TelemetryHandler(logger_provider=logger_provider)

    handler.guardrail_result("content_filter", "openai", triggered=False)

    assert len(log_exporter.get_finished_logs()) == expected_count


def test_guardrail_result_without_logger_provider_is_noop() -> None:
    handler = TelemetryHandler()

    handler.guardrail_result("content_filter", "openai", triggered=False)
