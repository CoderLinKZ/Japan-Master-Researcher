"""Non-serializable dependencies injected into graph nodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .ports import (
    CaseRepository,
    Clock,
    EventSink,
    IdGenerator,
    MCPGateway,
    MemoryStore,
    ModelRegistry,
    ObjectStore,
    RetryPolicy,
    SecurityPolicy,
    SystemClock,
    UUIDGenerator,
)


@dataclass(frozen=True, slots=True)
class JMRRuntimeContext:
    """Non-serializable production resources injected into graph nodes."""

    principal_user_id: str | None = None
    model_registry: ModelRegistry | None = None
    mcp_gateway: MCPGateway | None = None
    case_repository: CaseRepository | None = None
    memory_store: MemoryStore | None = None
    # The concrete LangGraph PostgresStore is kept separate from the typed
    # MemoryStore facade so nodes can use either a real store or a fake.
    postgres_store: Any | None = None
    object_store: ObjectStore | None = None
    clock: Clock = field(default_factory=SystemClock)
    id_generator: IdGenerator = field(default_factory=UUIDGenerator)
    retry_policy: RetryPolicy | None = None
    security_policy: SecurityPolicy | None = None
    event_sink: EventSink | None = None
    policies: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
