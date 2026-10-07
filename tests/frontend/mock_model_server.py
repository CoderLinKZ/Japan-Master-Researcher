"""Local browser-flow fixture: real API, database, and graph with a deterministic model.

Run only for interactive UI verification. It does not replace production model calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import jmr.api.service as api_service
from jmr.api import create_app
from jmr.api.auth import BearerTokenAuthenticator
from tests.fixtures.jmr_runtime import (
    DeterministicModelRegistry,
    FakeRetrievalGateway,
)


class BrowserTestModelRegistry(DeterministicModelRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(client=SimpleNamespace(close=lambda: None))


api_service.create_anthropic_model_registry = (
    lambda **_kwargs: BrowserTestModelRegistry()
)
api_service.NodeScopedMCPGateway = lambda *_args, **_kwargs: FakeRetrievalGateway()

app = create_app(
    service=api_service.ProductionAgentAPIService(),
    authenticator=BearerTokenAuthenticator(
        {
            "local-ui-test-token-for-root": "root",
            "local-ui-test-token-for-e2e": "ui-e2e",
        }
    ),
)
