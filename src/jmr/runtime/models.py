"""Production Anthropic adapters for structured and tool-calling nodes."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .ports import ModelToolCall, ToolModelTurn
from .reliability import BoundedRetryPolicy, call_with_retry
from .schema_validation import validate_json_schema

_STRUCTURED_TOOL = "submit_structured_result"
_REACT_FINAL_TOOL = "submit_retrieval_result"


class AnthropicNodeModel:
    """Translate Anthropic content blocks into provider-neutral model ports."""

    def __init__(
        self,
        client: Any,
        *,
        model_id: str,
        max_tokens: int = 8_000,
        retry_policy: BoundedRetryPolicy | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens < 256:
            raise ValueError("max_tokens must be at least 256")
        self.client = client
        self.model_id = model_id.strip()
        self.max_tokens = max_tokens
        self.retry_policy = retry_policy or BoundedRetryPolicy()

    def invoke_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        output_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        response = self._create(
            messages=messages,
            system_prompt=system_prompt,
            tools=[
                {
                    "name": _STRUCTURED_TOOL,
                    "description": "Submit the required schema-constrained result.",
                    "input_schema": dict(output_schema),
                }
            ],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL},
        )
        blocks = _tool_use_blocks(response)
        if len(blocks) != 1 or _field(blocks[0], "name") != _STRUCTURED_TOOL:
            raise ValueError("model did not return the required structured result")
        payload = _field(blocks[0], "input")
        if not isinstance(payload, Mapping):
            raise TypeError("structured model result must be an object")
        validate_json_schema(payload, output_schema)
        return dict(payload)

    def invoke_tool_step(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any],
    ) -> ToolModelTurn:
        advertised = [dict(tool) for tool in tools]
        if any(tool.get("name") == _REACT_FINAL_TOOL for tool in advertised):
            raise ValueError("reserved final-result tool name is already in use")
        advertised.append(
            {
                "name": _REACT_FINAL_TOOL,
                "description": (
                    "Finish this bounded retrieval worker with the exact structured "
                    "selection requested by the output schema."
                ),
                "input_schema": dict(output_schema),
            }
        )
        response = self._create(
            messages=messages,
            system_prompt=system_prompt,
            tools=advertised,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        blocks = _tool_use_blocks(response)
        finals = [
            block for block in blocks if _field(block, "name") == _REACT_FINAL_TOOL
        ]
        calls = [
            block for block in blocks if _field(block, "name") != _REACT_FINAL_TOOL
        ]
        if finals:
            if len(finals) != 1 or calls:
                raise ValueError("model mixed final output with MCP tool requests")
            payload = _field(finals[0], "input")
            if not isinstance(payload, Mapping):
                raise TypeError("final retrieval result must be an object")
            validate_json_schema(payload, output_schema)
            return ToolModelTurn(final_result=dict(payload))
        if not calls:
            raise ValueError(
                "ReAct model returned neither tool calls nor a final result"
            )
        schemas = {
            str(tool.get("name")): tool.get("input_schema", {}) for tool in advertised
        }
        normalized_calls: list[ModelToolCall] = []
        for block in calls:
            name = str(_field(block, "name") or "")
            arguments = _mapping(_field(block, "input"), "tool input")
            schema = schemas.get(name)
            if not isinstance(schema, Mapping):
                raise ValueError(f"model requested an unknown tool: {name!r}")
            # Some Anthropic-compatible endpoints add a dummy field to a
            # no-argument tool call. These tools receive host-built arguments
            # only, so discard the model's entire payload before validation.
            if (
                schema.get("type") == "object"
                and schema.get("properties") == {}
                and schema.get("additionalProperties") is False
                and not schema.get("required")
            ):
                arguments = {}
            validate_json_schema(arguments, schema)
            normalized_calls.append(
                ModelToolCall(
                    call_id=str(_field(block, "id") or ""),
                    name=name,
                    arguments=arguments,
                )
            )
        return ToolModelTurn(tool_calls=tuple(normalized_calls))

    def _create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: Mapping[str, Any],
    ) -> Any:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        request = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [dict(message) for message in messages],
            "tools": [dict(tool) for tool in tools],
            "tool_choice": dict(tool_choice),
        }
        return call_with_retry(
            lambda: self.client.messages.create(**request),
            policy=self.retry_policy,
            idempotent=True,
        )


class AnthropicModelRegistry:
    """Use one configured Anthropic adapter with node-local prompts/tools."""

    def __init__(self, model: AnthropicNodeModel) -> None:
        if not isinstance(model, AnthropicNodeModel):
            raise TypeError("model must be an AnthropicNodeModel")
        self.model = model

    def for_node(self, node_name: str) -> AnthropicNodeModel:
        if not isinstance(node_name, str) or not node_name.strip():
            raise ValueError("node_name must be non-empty")
        return self.model


def create_anthropic_model_registry(
    *,
    environment: Mapping[str, str] | None = None,
    retry_policy: BoundedRetryPolicy | None = None,
) -> AnthropicModelRegistry:
    """Build the production registry from explicit environment settings."""

    selected = os.environ if environment is None else environment
    api_key = selected.get("ANTHROPIC_API_KEY", "").strip()
    model_id = selected.get("MODEL_ID", "").strip()
    if not api_key or not model_id:
        missing = [
            name
            for name, value in (
                ("ANTHROPIC_API_KEY", api_key),
                ("MODEL_ID", model_id),
            )
            if not value
        ]
        raise RuntimeError(
            "Missing required model configuration: " + ", ".join(missing)
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("anthropic dependency is not installed") from exc
    timeout_raw = selected.get("MODEL_TIMEOUT_SECONDS", "60").strip()
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError("MODEL_TIMEOUT_SECONDS must be numeric") from exc
    if not 1 <= timeout_seconds <= 120:
        raise RuntimeError("MODEL_TIMEOUT_SECONDS must be between 1 and 120")
    arguments: dict[str, Any] = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": timeout_seconds,
    }
    base_url = selected.get("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        arguments["base_url"] = base_url
    client = Anthropic(**arguments)
    return AnthropicModelRegistry(
        AnthropicNodeModel(
            client,
            model_id=model_id,
            retry_policy=retry_policy,
        )
    )


def _tool_use_blocks(response: Any) -> list[Any]:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        raise TypeError("Anthropic response content must be a list")
    return [block for block in content if _field(block, "type") == "tool_use"]


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return dict(value)


__all__ = [
    "AnthropicModelRegistry",
    "AnthropicNodeModel",
    "create_anthropic_model_registry",
]
