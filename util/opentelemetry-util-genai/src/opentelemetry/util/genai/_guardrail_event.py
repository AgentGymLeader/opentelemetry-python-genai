# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.util.genai.utils import should_emit_guardrail_result_event

# Placeholder until semconv-genai#427 defines the guardrail result event; at
# fffe24ae it defines only gen_ai.guardrail.security.finding.
_GUARDRAIL_RESULT_EVENT_NAME = "gen_ai.guardrail.result"

# semconv-genai#427 (unreleased)
# Switch to the semconv module once semconv-genai#427 merges.
_GEN_AI_GUARDRAIL_COMPONENT_NAME = "gen_ai.guardrail.component.name"
# semconv-genai#427 (unreleased)
# Switch to the semconv module once semconv-genai#427 merges.
_GEN_AI_GUARDRAIL_VERDICT_TYPE = "gen_ai.guardrail.verdict.type"


def emit_guardrail_result(
    logger: Logger,
    name: str,
    provider: str,
    *,
    triggered: bool,
    error_type: str | None = None,
) -> None:
    if not should_emit_guardrail_result_event():
        return
    attributes = {
        _GEN_AI_GUARDRAIL_COMPONENT_NAME: name,
        GenAI.GEN_AI_PROVIDER_NAME: provider,
    }
    # gen_ai.guardrail.target.type is Required in #427, but openai-agents span
    # data does not record whether an input or output guardrail ran.
    if error_type is not None:
        attributes[ERROR_TYPE] = error_type
    else:
        # The OpenAI Agents SDK exposes only a binary tripwire, so this is an
        # exact mapping, not a flattening of a richer verdict.
        attributes[_GEN_AI_GUARDRAIL_VERDICT_TYPE] = (
            "deny" if triggered else "allow"
        )
    logger.emit(
        LogRecord(
            event_name=_GUARDRAIL_RESULT_EVENT_NAME,
            attributes=attributes,
            context=get_current(),
        )
    )
