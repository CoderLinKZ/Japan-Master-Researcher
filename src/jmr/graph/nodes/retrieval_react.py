"""Strict bounded ReAct runtime for the two evidence-retrieval workers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import quote

from jmr.domain import MCPStatus, RetrievalResult
from jmr.graph.manifest import agent_tools_for
from jmr.runtime import JMRRuntimeContext, ToolModelTurn

from .common import emit_event, repository, required_text

MAX_REACT_ROUNDS = 6
REPEATED_TOOL_ERROR_LIMIT = 3

_PUBLICATION_NODE = "publication_research_agent"
_KAKEN_NODE = "kaken_research_agent"
_DISCOVERY_TOOL = "mcp__scholar__discover_official_sources"
_PUBLICATION_TOOL = "mcp__scholar__search_publications"


def run_bounded_react_worker(
    state: Mapping[str, Any],
    context: JMRRuntimeContext,
    *,
    worker_kind: str,
) -> dict[str, Any]:
    """Run, validate, persist, and return one bounded ReAct worker result."""

    if worker_kind not in {"publication", "kaken"}:
        raise ValueError("worker_kind must be publication or kaken")
    repo = repository(context)
    user_id = required_text(state.get("user_id"), "user_id")
    case_id = required_text(state.get("case_id"), "case_id")
    case = repo.get_case(user_id=user_id, case_id=case_id)
    target = dict((case or {}).get("target") or {})
    if not target:
        raise RuntimeError("ReAct retrieval requires a persisted target")
    saved_plan = repo.get_research_plan(
        user_id=user_id,
        case_id=case_id,
        research_plan_id=required_text(
            state.get("research_plan_id"), "research_plan_id"
        ),
    )
    plan = saved_plan.get("plan", {})
    model_plan = plan.get("model_plan", {}) if isinstance(plan, Mapping) else {}

    node_name = _PUBLICATION_NODE if worker_kind == "publication" else _KAKEN_NODE
    model = _require_tool_model(context, node_name)
    tools = _node_tools(context, node_name)
    messages: list[Mapping[str, Any]] = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "retrieve_evidence",
                    "worker_kind": worker_kind,
                    "research_plan_id": state.get("research_plan_id"),
                    "planned_queries": _planned_queries(model_plan, worker_kind),
                    "planned_stop_conditions": _plan_strings(
                        model_plan, "stop_conditions"
                    ),
                    "target": _safe_target(target),
                    "max_react_rounds": MAX_REACT_ROUNDS,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    ]
    observed_results: list[RetrievalResult] = []
    observed_source_objects: dict[str, dict[str, Any]] = {}
    observed_sources: dict[str, dict[str, Any]] = {}
    latest_result: RetrievalResult | None = None
    last_error_signature: str | None = None
    repeated_error_count = 0
    invalid_final_count = 0
    raw_result: dict[str, Any] | None = None

    for round_number in range(1, MAX_REACT_ROUNDS + 1):
        try:
            turn = model.invoke_tool_step(
                messages=tuple(messages),
                system_prompt=_worker_prompt(worker_kind),
                tools=tools,
                output_schema=_final_selection_schema(),
            )
        except Exception as exc:
            raw_result = _failed_result(
                context,
                worker_kind,
                f"model step failed with {type(exc).__name__}",
            )
            emit_event(
                context,
                state,
                node=node_name,
                status="FAILED",
                details={"failure_type": type(exc).__name__},
            )
            break
        if not isinstance(turn, ToolModelTurn):
            raw_result = _failed_result(
                context,
                worker_kind,
                "model step returned an invalid turn type",
            )
            break

        if turn.final_result is not None:
            try:
                final = _grounded_final_result(
                    turn.final_result,
                    worker_kind=worker_kind,
                    observed_results=observed_results,
                )
            except (TypeError, ValueError) as exc:
                invalid_final_count += 1
                if invalid_final_count >= 2:
                    raw_result = _failed_result(
                        context,
                        worker_kind,
                        "model returned an invalid or ungrounded final result twice",
                    )
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "type": "final_result_validation_error",
                                "error": type(exc).__name__,
                                "instruction": (
                                    "Return only the request_id of one previously "
                                    "observed RetrievalResult."
                                ),
                            },
                            sort_keys=True,
                        ),
                    }
                )
                continue
            raw_result = final.to_dict()
            break

        if len(turn.tool_calls) != 1:
            raw_result = _failed_result(
                context,
                worker_kind,
                "retrieval workers require exactly one tool call per ReAct round",
            )
            break

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                    for call in turn.tool_calls
                ],
            }
        )
        terminate = False
        exposed_names = {tool["name"] for tool in tools}
        for call in turn.tool_calls:
            if call.name not in exposed_names:
                raw_result = _failed_result(
                    context,
                    worker_kind,
                    f"model requested a tool outside the {node_name} allowlist",
                )
                terminate = True
                break
            arguments = _bounded_arguments(
                worker_kind=worker_kind,
                tool_name=call.name,
                target=target,
                observed_sources=observed_sources,
            )
            emit_event(
                context,
                state,
                node=node_name,
                status="STARTED",
                tool=call.name,
                details={"tool": call.name, "attempt": round_number},
            )
            tool_started = time.monotonic()
            try:
                raw = context.mcp_gateway.call_tool(
                    node_name=node_name,
                    tool_name=call.name,
                    arguments=arguments,
                    trusted_context={"user_id": user_id, "case_id": case_id},
                )
                observation = _unwrap_result(raw)
                source_object = _snapshot_tool_response(
                    context,
                    user_id=user_id,
                    case_id=case_id,
                    tool_name=call.name,
                    response=raw,
                )
                status = _mcp_status(observation.get("status"))
                _record_discovered_sources(observation, observed_sources)
                retrieval = _as_retrieval_result(
                    observation,
                    worker_kind=worker_kind,
                    tool_name=call.name,
                )
                if retrieval is not None:
                    if source_object is not None:
                        retrieval = replace(
                            retrieval,
                            artifact_ref=source_object["object_uri"],
                        )
                        observed_source_objects[retrieval.request_id] = source_object
                    observed_results.append(retrieval)
                    latest_result = retrieval
                messages.append(_tool_result_message(call.call_id, observation))

                if status is MCPStatus.NEEDS_USER_CONFIRMATION:
                    raw_result = (
                        retrieval.to_dict()
                        if retrieval is not None
                        else _discovery_terminal_result(
                            context, observation, status=status
                        )
                    )
                    if retrieval is None and source_object is not None:
                        raw_result["artifact_ref"] = source_object["object_uri"]
                        observed_source_objects[raw_result["request_id"]] = (
                            source_object
                        )
                    _emit_tool_outcome(
                        context,
                        state,
                        node_name=node_name,
                        tool_name=call.name,
                        event_status="SUCCEEDED",
                        mcp_status=status,
                        attempt=round_number,
                        duration_ms=(time.monotonic() - tool_started) * 1000,
                    )
                    terminate = True
                    break
                if status in {
                    MCPStatus.NO_RESULT,
                    MCPStatus.BLOCKED,
                    MCPStatus.CONFLICT,
                }:
                    raw_result = (
                        retrieval.to_dict()
                        if retrieval is not None
                        else _discovery_terminal_result(
                            context, observation, status=status
                        )
                    )
                    if retrieval is None and source_object is not None:
                        raw_result["artifact_ref"] = source_object["object_uri"]
                        observed_source_objects[raw_result["request_id"]] = (
                            source_object
                        )
                    _emit_tool_outcome(
                        context,
                        state,
                        node_name=node_name,
                        tool_name=call.name,
                        event_status="SUCCEEDED",
                        mcp_status=status,
                        attempt=round_number,
                        duration_ms=(time.monotonic() - tool_started) * 1000,
                    )
                    terminate = True
                    break
                if status is MCPStatus.FAILED:
                    error_signature = _tool_error_signature(
                        call.name,
                        arguments,
                        _result_error(observation),
                    )
                    last_error_signature, repeated_error_count = _advance_error_counter(
                        previous=last_error_signature,
                        count=repeated_error_count,
                        current=error_signature,
                    )
                    is_open = repeated_error_count >= REPEATED_TOOL_ERROR_LIMIT
                    _emit_tool_outcome(
                        context,
                        state,
                        node_name=node_name,
                        tool_name=call.name,
                        event_status="FAILED" if is_open else "RETRYING",
                        mcp_status=status,
                        attempt=round_number,
                        retry_count=repeated_error_count,
                        duration_ms=(time.monotonic() - tool_started) * 1000,
                    )
                    if is_open:
                        raw_result = _failed_result(
                            context,
                            worker_kind,
                            "repeated identical tool error circuit breaker opened",
                        )
                        terminate = True
                        break
                else:
                    last_error_signature = None
                    repeated_error_count = 0
                    _emit_tool_outcome(
                        context,
                        state,
                        node_name=node_name,
                        tool_name=call.name,
                        event_status="SUCCEEDED",
                        mcp_status=status,
                        attempt=round_number,
                        duration_ms=(time.monotonic() - tool_started) * 1000,
                    )
            except Exception as exc:  # external boundary becomes an observation
                error_signature = _tool_error_signature(
                    call.name,
                    arguments,
                    f"{type(exc).__name__}:{exc}",
                )
                last_error_signature, repeated_error_count = _advance_error_counter(
                    previous=last_error_signature,
                    count=repeated_error_count,
                    current=error_signature,
                )
                messages.append(
                    _tool_result_message(
                        call.call_id,
                        {
                            "status": MCPStatus.FAILED.value,
                            "error_type": type(exc).__name__,
                            "retry_count": repeated_error_count,
                        },
                    )
                )
                is_open = repeated_error_count >= REPEATED_TOOL_ERROR_LIMIT
                _emit_tool_outcome(
                    context,
                    state,
                    node_name=node_name,
                    tool_name=call.name,
                    event_status="FAILED" if is_open else "RETRYING",
                    mcp_status=MCPStatus.FAILED,
                    attempt=round_number,
                    retry_count=repeated_error_count,
                    duration_ms=(time.monotonic() - tool_started) * 1000,
                )
                if is_open:
                    raw_result = _failed_result(
                        context,
                        worker_kind,
                        "repeated identical tool error circuit breaker opened",
                    )
                    terminate = True
                    break
        if terminate:
            break

    if raw_result is None:
        raw_result = _loop_limit_result(context, worker_kind, latest_result)

    status = _mcp_status(raw_result.get("status"))
    saved = repo.save_retrieval_run(
        user_id=user_id,
        case_id=case_id,
        worker_kind=worker_kind,
        status=status.value,
        result_summary=raw_result,
        source_object=observed_source_objects.get(raw_result.get("request_id")),
        idempotency_key=(
            f"retrieval:{case_id}:{worker_kind}:{state.get('research_plan_id')}"
        ),
    )
    emit_event(context, state, node=node_name, status=status.value)
    return {
        "retrieval_refs": [
            {
                "worker_kind": worker_kind,
                "retrieval_run_id": saved["retrieval_run_id"],
                "mcp_status": status.value,
            }
        ],
        "warnings": _string_list(raw_result.get("warnings")),
        "errors": _string_list(raw_result.get("errors")),
    }


def _snapshot_tool_response(
    context: JMRRuntimeContext,
    *,
    user_id: str,
    case_id: str,
    tool_name: str,
    response: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Persist the complete MCP observation outside checkpoint/business JSON.

    The current MCP servers expose normalized API/HTML extraction responses;
    binary upstream pages are not fabricated by this snapshot boundary.
    """

    if context.object_store is None:
        return None
    encoded = json.dumps(
        response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    if len(encoded) > 8 * 1024 * 1024:
        raise ValueError("MCP response exceeds the 8 MiB source snapshot limit")
    digest = hashlib.sha256(encoded).hexdigest()
    object_key = (
        f"users/{quote(user_id, safe='')}/cases/{quote(case_id, safe='')}/"
        f"sources/{digest}.json"
    )
    uri = context.object_store.put(
        object_key=object_key,
        content=encoded,
        content_type="application/json",
    )
    return {
        "object_uri": uri,
        "sha256": digest,
        "mime_type": "application/json",
        "byte_size": len(encoded),
        "fetched_at": context.clock.now().isoformat(),
        "metadata": {
            "capture_type": "mcp_tool_response",
            "tool_name": tool_name,
        },
    }


def _require_tool_model(context: JMRRuntimeContext, node_name: str) -> Any:
    if context.model_registry is None:
        raise RuntimeError(f"{node_name} requires a tool-calling model registry")
    model = context.model_registry.for_node(node_name)
    if not callable(getattr(model, "invoke_tool_step", None)):
        raise RuntimeError(f"{node_name} requires a ToolCallingModel")
    return model


def _node_tools(
    context: JMRRuntimeContext, node_name: str
) -> tuple[dict[str, Any], ...]:
    if context.mcp_gateway is None:
        raise RuntimeError(f"{node_name} requires an MCP gateway")
    expected = set(agent_tools_for(node_name))
    exposed: dict[str, dict[str, Any]] = {}
    for raw in context.mcp_gateway.tools_for_node(node_name):
        if not isinstance(raw, Mapping):
            raise TypeError("MCP tool definitions must be objects")
        name = raw.get("name")
        if isinstance(name, str) and name in expected:
            exposed[name] = dict(raw)
    missing = expected - set(exposed)
    if missing:
        raise RuntimeError(
            f"{node_name} is missing allowlisted MCP tools: "
            + ", ".join(sorted(missing))
        )
    return tuple(exposed[name] for name in agent_tools_for(node_name))


def _bounded_arguments(
    *,
    worker_kind: str,
    tool_name: str,
    target: Mapping[str, Any],
    observed_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build Tool arguments exclusively from the persisted Case boundary."""

    arguments: dict[str, Any] = {
        "university_name": required_text(
            target.get("university_name"), "university_name"
        ),
        "graduate_school_name": required_text(
            target.get("graduate_school_name"), "graduate_school_name"
        ),
        "professor_name": required_text(target.get("professor_name"), "professor_name"),
        "professor_name_variants": _string_list(target.get("professor_name_variants")),
        "search_start_date": required_text(target.get("date_from"), "date_from"),
        "search_end_date": required_text(target.get("date_to"), "date_to"),
        "max_results": 50,
    }
    if worker_kind == "publication":
        laboratory = target.get("laboratory_name")
        if isinstance(laboratory, str) and laboratory.strip():
            arguments["laboratory_name"] = laboratory.strip()
        arguments["official_urls"] = _string_list(target.get("official_urls"))
        if tool_name == _PUBLICATION_TOOL and observed_sources:
            arguments["official_sources"] = [
                dict(source) for source in observed_sources.values()
            ]
        if tool_name == _DISCOVERY_TOOL:
            arguments["max_results"] = 20
    return arguments


def _grounded_final_result(
    value: Mapping[str, Any],
    *,
    worker_kind: str,
    observed_results: Sequence[RetrievalResult],
) -> RetrievalResult:
    if set(value) != {"request_id"}:
        raise ValueError("final result must contain only request_id")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("final request_id must be non-empty")
    result = next(
        (item for item in observed_results if item.request_id == request_id),
        None,
    )
    if result is None:
        raise ValueError("final result must select an observed tool result")
    expected_type = "publication" if worker_kind == "publication" else "kaken_project"
    if result.evidence_type.value != expected_type:
        raise ValueError("final result has the wrong evidence_type")
    return result


def _as_retrieval_result(
    observation: Mapping[str, Any],
    *,
    worker_kind: str,
    tool_name: str,
) -> RetrievalResult | None:
    if tool_name == _DISCOVERY_TOOL:
        return None
    try:
        result = RetrievalResult.from_dict(observation)
    except (TypeError, ValueError) as exc:
        raise ValueError("retrieval MCP returned an invalid RetrievalResult") from exc
    expected_type = "publication" if worker_kind == "publication" else "kaken_project"
    if result.evidence_type.value != expected_type:
        raise ValueError("retrieval MCP returned the wrong evidence_type")
    return result


def _record_discovered_sources(
    observation: Mapping[str, Any],
    target: dict[str, dict[str, Any]],
) -> None:
    raw_sources = observation.get("sources", observation.get("discovered_sources", []))
    if not isinstance(raw_sources, list):
        return
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            continue
        url = raw.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            target[url] = dict(raw)


def _discovery_terminal_result(
    context: JMRRuntimeContext,
    observation: Mapping[str, Any],
    *,
    status: MCPStatus,
) -> dict[str, Any]:
    normalized_status = MCPStatus.FAILED if status is MCPStatus.CONFLICT else status
    return {
        "request_id": observation.get("request_id")
        or context.id_generator.new_id("request"),
        "evidence_type": "publication",
        "status": normalized_status.value,
        "query": observation.get("query") or "official source discovery",
        "source_name": observation.get("source_name") or "scholar",
        "retrieved_at": observation.get("retrieved_at")
        or context.clock.now().isoformat(),
        "records": [],
        "discovered_sources": list(observation.get("sources") or []),
        "warnings": _string_list(observation.get("warnings")),
        "errors": _string_list(observation.get("errors")),
        "artifact_ref": None,
    }


def _loop_limit_result(
    context: JMRRuntimeContext,
    worker_kind: str,
    latest: RetrievalResult | None,
) -> dict[str, Any]:
    if latest is None or not latest.records:
        return _failed_result(
            context,
            worker_kind,
            f"bounded ReAct worker reached {MAX_REACT_ROUNDS} rounds without evidence",
        )
    payload = latest.to_dict()
    payload["status"] = MCPStatus.PARTIAL.value
    payload["warnings"] = list(
        dict.fromkeys(
            [
                *_string_list(payload.get("warnings")),
                (f"bounded ReAct worker stopped at the {MAX_REACT_ROUNDS}-round limit"),
            ]
        )
    )
    return RetrievalResult.from_dict(payload).to_dict()


def _failed_result(
    context: JMRRuntimeContext, worker_kind: str, error: str
) -> dict[str, Any]:
    return {
        "request_id": context.id_generator.new_id("request"),
        "evidence_type": (
            "publication" if worker_kind == "publication" else "kaken_project"
        ),
        "status": MCPStatus.FAILED.value,
        "query": f"worker={worker_kind}",
        "source_name": "unavailable",
        "retrieved_at": context.clock.now().isoformat(),
        "records": [],
        "discovered_sources": [],
        "warnings": [],
        "errors": [error],
        "artifact_ref": None,
    }


def _unwrap_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("MCP response must be an object")
    if isinstance(raw.get("data"), Mapping):
        data = dict(raw["data"])
        data.setdefault("status", raw.get("status"))
        return data
    return dict(raw)


def _mcp_status(value: Any) -> MCPStatus:
    normalized = str(value or "FAILED").upper()
    try:
        return MCPStatus(normalized)
    except ValueError:
        return MCPStatus.FAILED


def _result_error(observation: Mapping[str, Any]) -> str:
    errors = _string_list(observation.get("errors"))
    return "|".join(errors) if errors else str(observation.get("status", "FAILED"))


def _tool_error_signature(
    tool_name: str, arguments: Mapping[str, Any], error: str
) -> str:
    return _canonical_json(
        {"tool": tool_name, "arguments": dict(arguments), "error": error}
    )


def _advance_error_counter(
    *, previous: str | None, count: int, current: str
) -> tuple[str, int]:
    return current, count + 1 if previous == current else 1


def _tool_result_message(call_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    records = result.get("records")
    sources = result.get("sources", result.get("discovered_sources"))
    compact = {
        "request_id": result.get("request_id"),
        "status": str(result.get("status", MCPStatus.FAILED.value)),
        "record_count": len(records) if isinstance(records, list) else 0,
        "evidence_ids": [
            item.get("evidence_id")
            for item in (records if isinstance(records, list) else [])
            if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str)
        ],
        "record_previews": [
            {
                "evidence_id": item.get("evidence_id"),
                "title": _preview_text(item.get("title")),
                "authors": [
                    _preview_text(value)
                    for value in _string_list(item.get("authors"))[:6]
                ],
                "affiliations": [
                    _preview_text(value)
                    for value in _string_list(item.get("affiliations"))[:4]
                ],
                "publication_date": _preview_text(item.get("publication_date")),
                "verification_status": item.get("verification_status"),
            }
            for item in (records[:12] if isinstance(records, list) else [])
            if isinstance(item, Mapping)
        ],
        "record_previews_truncated": isinstance(records, list) and len(records) > 12,
        "source_count": len(sources) if isinstance(sources, list) else 0,
        "warnings": _string_list(result.get("warnings"))[:8],
        "errors": _string_list(result.get("errors"))[:8],
    }
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": json.dumps(
                    compact, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
        ],
    }


def _preview_text(value: Any) -> str:
    return value.strip()[:160] if isinstance(value, str) else ""


def _emit_tool_outcome(
    context: JMRRuntimeContext,
    state: Mapping[str, Any],
    *,
    node_name: str,
    tool_name: str,
    event_status: str,
    mcp_status: MCPStatus,
    attempt: int,
    retry_count: int = 0,
    duration_ms: float = 0.0,
) -> None:
    details: dict[str, Any] = {
        "tool": tool_name,
        "attempt": attempt,
        "mcp_status": mcp_status.value,
    }
    if retry_count:
        details["retry_count"] = retry_count
    emit_event(
        context,
        state,
        node=node_name,
        status=event_status,
        tool=tool_name,
        duration_ms=duration_ms,
        retry_count=retry_count,
        details=details,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _safe_target(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: target.get(key)
        for key in (
            "university_name",
            "graduate_school_name",
            "laboratory_name",
            "professor_name",
            "professor_name_variants",
            "official_urls",
            "date_from",
            "date_to",
        )
    }


def _plan_strings(plan: Any, field: str) -> list[str]:
    if not isinstance(plan, Mapping) or not isinstance(plan.get(field), list):
        return []
    return [
        value.strip()[:160]
        for value in plan[field][:4]
        if isinstance(value, str) and value.strip()
    ]


def _planned_queries(plan: Any, worker_kind: str) -> list[str]:
    field = "publication_queries" if worker_kind == "publication" else "kaken_queries"
    return _plan_strings(plan, field)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _worker_prompt(worker_kind: str) -> str:
    common = (
        "You are a bounded ReAct evidence-retrieval worker. Use only the tools "
        "provided to this node. The host enforces professor, institution, date "
        "range, and result limits, so call each retrieval tool with an empty "
        "input object. Use the persisted plan's queries and stop conditions "
        "only to guide tool choice and stopping; they are not evidence. "
        "Never invent or alter a record, identifier, "
        "source, warning, or error. On completion, final_result must contain only "
        "the request_id of a RetrievalResult previously returned by a tool. "
        "Return tool "
        "calls or final_result, never both. NEEDS_USER_CONFIRMATION is terminal."
    )
    if worker_kind == "publication":
        return (
            common + " Discover official sources only when confirmed URLs are "
            "insufficient, then search publications. Check identity, institution, "
            "author order, dates, source type, and truncation warnings."
        )
    return (
        common + " Search KAKEN projects and check project number, member identity, "
        "institution, role, active years, and official source. Do not apply "
        "publication concepts such as author order or venue ranking."
    )


def _final_selection_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"request_id": {"type": "string", "minLength": 1}},
        "required": ["request_id"],
    }
