"""Production CLI for the single compiled JMR workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from jmr.graph import GRAPH_STATE_SCHEMA_VERSION, compile_graph, create_initial_state
from jmr.persistence import (
    FileObjectStore,
    PostgresMemoryStore,
    open_postgres_repository,
    open_postgres_store,
)
from jmr.runtime import (
    BoundedRetryPolicy,
    CheckpointerConfigurationError,
    JMRRuntimeContext,
    JsonLineEventSink,
    MCPServerRegistry,
    NodeScopedMCPGateway,
    create_anthropic_model_registry,
    new_run_id,
    open_postgres_checkpointer,
)
from jmr.runtime.security import PrincipalSecurityPolicy
from mcp_servers import create_kaken_server, create_scholar_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    normalized = value.strip()
    if len(normalized) > 255:
        raise ValueError(f"{field_name} cannot exceed 255 characters")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Japan Master Researcher through its production graph.",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("JMR_USER_ID"),
        help="Authenticated, stable user identifier (or JMR_USER_ID).",
    )
    parser.add_argument(
        "--case-id",
        help="Stable case/thread identifier used to resume checkpoints.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Apply checkpoint, Store, and JMR business migrations at startup.",
    )
    return parser.parse_args(argv)


def load_runtime_environment() -> None:
    """Load unset values from the untracked project environment file."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


def validate_model_environment() -> None:
    missing = [
        name
        for name in ("ANTHROPIC_API_KEY", "MODEL_ID")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


def run_turn(
    graph: Any,
    *,
    user_id: str,
    case_id: str,
    runtime_context: JMRRuntimeContext,
    user_input: str | None = None,
    resume: Mapping[str, Any] | None = None,
    resume_interrupt_id: str | None = None,
) -> dict[str, Any]:
    """Invoke or resume one case while enforcing interrupt semantics."""

    normalized_user_id = _identifier(user_id, "user_id")
    normalized_case_id = _identifier(case_id, "case_id")
    config = {
        "configurable": {
            "user_id": normalized_user_id,
            "thread_id": normalized_case_id,
        }
    }
    principal = runtime_context.principal_user_id
    if runtime_context.security_policy is not None:
        if not isinstance(principal, str) or not principal.strip():
            raise PermissionError("an authenticated principal is required")
        runtime_context.security_policy.authorize_case(
            principal_user_id=principal,
            user_id=normalized_user_id,
            case_id=normalized_case_id,
        )
    elif isinstance(principal, str) and principal.strip() != normalized_user_id:
        raise PermissionError("principal does not own this user namespace")
    if runtime_context.case_repository is not None:
        # Resolve ownership from the business authority before touching a
        # checkpoint selected by caller-controlled thread_id.
        runtime_context.case_repository.get_case(
            user_id=normalized_user_id,
            case_id=normalized_case_id,
        )
    snapshot = graph.get_state(config)
    if snapshot.values:
        checkpoint_user = _identifier(snapshot.values.get("user_id"), "state.user_id")
        checkpoint_case = _identifier(snapshot.values.get("case_id"), "state.case_id")
        if (
            checkpoint_user != normalized_user_id
            or checkpoint_case != normalized_case_id
        ):
            raise PermissionError("checkpoint identity does not match this invocation")
        if snapshot.values.get("schema_version") != GRAPH_STATE_SCHEMA_VERSION:
            raise ValueError("checkpoint graph schema version is unsupported")
    has_interrupt = bool(snapshot.tasks) and any(
        getattr(task, "interrupts", ()) for task in snapshot.tasks
    )
    turn_context = replace(
        runtime_context,
        metadata={**runtime_context.metadata, "run_id": new_run_id()},
    )
    if resume is not None:
        if not has_interrupt:
            raise ValueError("resume payload supplied but no interrupt is pending")
        resume_value: dict[str, Any] = dict(resume)
        if resume_interrupt_id is not None:
            pending_ids = {
                interrupt.id
                for task in snapshot.tasks
                for interrupt in getattr(task, "interrupts", ())
                if isinstance(getattr(interrupt, "id", None), str)
            }
            if resume_interrupt_id not in pending_ids:
                raise ValueError("the requested interrupt is no longer pending")
            resume_value = {resume_interrupt_id: resume_value}
        return dict(
            graph.invoke(
                Command(resume=resume_value),
                config=config,
                context=turn_context,
            )
        )
    if has_interrupt:
        raise ValueError("an interrupt is pending; provide a JSON resume payload")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input is required when resume is absent")
    graph_input = (
        {"messages": [{"role": "user", "content": user_input.strip()}]}
        if snapshot.values
        else create_initial_state(
            normalized_user_id,
            normalized_case_id,
            messages=[{"role": "user", "content": user_input.strip()}],
        )
    )
    return dict(graph.invoke(graph_input, config=config, context=turn_context))


def _configure_console_encoding() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def _has_pending_interrupt(graph: Any, config: Mapping[str, Any]) -> bool:
    snapshot = graph.get_state(config)
    return bool(snapshot.tasks) and any(
        getattr(task, "interrupts", ()) for task in snapshot.tasks
    )


def _interrupt_payload(result: Mapping[str, Any]) -> Any:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def _print_result(result: Mapping[str, Any], runtime: JMRRuntimeContext) -> None:
    payload = _interrupt_payload(result)
    if payload is not None:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    iteration_id = result.get("current_outreach_iteration_id")
    if result.get("run_status") == "COMPLETED" and isinstance(iteration_id, str):
        repository = runtime.case_repository
        if repository is not None:
            iteration = repository.get_outreach_iteration(
                user_id=result["user_id"],
                case_id=result["case_id"],
                iteration_id=iteration_id,
            )
            print(iteration.get("content", "(completed without printable content)"))
            return
    print(
        json.dumps(
            {
                "workflow_stage": result.get("workflow_stage"),
                "run_status": result.get("run_status"),
                "warnings": result.get("warnings", []),
                "errors": result.get("errors", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a durable interactive case with production dependencies."""

    _configure_console_encoding()
    load_runtime_environment()
    args = parse_args(argv)
    try:
        user_id = _identifier(args.user_id, "user_id")
        case_id = _identifier(args.case_id or str(uuid.uuid4()), "case_id")
        validate_model_environment()
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"Startup error: {exc}")
        return 1

    object_root = Path(
        os.getenv("JMR_OBJECT_STORE_DIR", str(PROJECT_ROOT / ".jmr" / "objects"))
    )
    event_path = Path(
        os.getenv("JMR_EVENT_LOG_PATH", str(PROJECT_ROOT / ".jmr" / "events.jsonl"))
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    print("Japan Master Researcher")
    print(f"User ID: {user_id}")
    print(f"Case ID: {case_id}")
    print("Interrupts require a JSON resume object. Type q or exit to quit.\n")

    try:
        with ExitStack() as stack:
            checkpointer = stack.enter_context(
                open_postgres_checkpointer(setup=args.setup)
            )
            repository = stack.enter_context(open_postgres_repository(setup=args.setup))
            store = stack.enter_context(open_postgres_store(setup=args.setup))
            event_stream = stack.enter_context(event_path.open("a", encoding="utf-8"))
            retry_policy = BoundedRetryPolicy()
            runtime = JMRRuntimeContext(
                principal_user_id=user_id,
                model_registry=create_anthropic_model_registry(
                    retry_policy=retry_policy
                ),
                mcp_gateway=NodeScopedMCPGateway(
                    MCPServerRegistry(
                        {
                            "scholar": create_scholar_server,
                            "kaken": create_kaken_server,
                        }
                    ),
                    retry_policy=retry_policy,
                    audit_writer=repository.record_audit_event,
                ),
                case_repository=repository,
                memory_store=PostgresMemoryStore(store),
                postgres_store=store,
                object_store=FileObjectStore(object_root),
                retry_policy=retry_policy,
                security_policy=PrincipalSecurityPolicy(),
                event_sink=JsonLineEventSink(event_stream),
                metadata={"run_id": new_run_id()},
            )
            graph = compile_graph(checkpointer)
            config = {"configurable": {"user_id": user_id, "thread_id": case_id}}
            while True:
                pending = _has_pending_interrupt(graph, config)
                try:
                    raw = input("Resume JSON > " if pending else "JMR > ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if raw.strip().lower() in {"q", "exit"}:
                    break
                if not raw.strip():
                    continue
                try:
                    result = run_turn(
                        graph,
                        user_id=user_id,
                        case_id=case_id,
                        runtime_context=runtime,
                        resume=json.loads(raw) if pending else None,
                        user_input=None if pending else raw,
                    )
                    _print_result(result, runtime)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    print(f"Input error: {exc}")
                except Exception as exc:
                    print(f"Workflow error: {type(exc).__name__}: {exc}")
    except (CheckpointerConfigurationError, OSError, RuntimeError) as exc:
        print(f"Startup error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
