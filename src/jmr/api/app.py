"""FastAPI application factory for JMR."""

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path as FilePath
from typing import Annotated, Any

from anthropic import APIConnectionError, APITimeoutError
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from psycopg import InterfaceError, OperationalError
from psycopg.errors import UndefinedTable

from jmr.graph.nodes.common import ModelOutputValidationError
from jmr.operations.health import liveness
from jmr.persistence import (
    ConstraintError,
    IdempotencyConflictError,
    NotFoundError,
    OwnershipError,
    RepositoryConfigurationError,
    StoreConfigurationError,
    VersionConflictError,
)
from jmr.runtime import CheckpointerConfigurationError, failure_reason_code

from .auth import BearerTokenAuthenticator
from .middleware import RequestBodyLimitMiddleware
from .models import (
    CaseDeletionView,
    CaseListView,
    CaseProgressView,
    CaseView,
    ConfirmCaseDeletionRequest,
    ConversationView,
    CreateCaseRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    ErrorView,
    FinalResultView,
    MemoryListView,
    MemoryView,
    ResumeCaseRequest,
    SendMessageRequest,
    UpdateMemoryRequest,
    WorkspaceView,
)
from .service import (
    AgentAPIService,
    CaseBusyError,
    OperationConflictError,
    ProductionAgentAPIService,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from .uploads import MAX_FILE_BYTES, UploadValidationError

CASE_ID_PATH = Path(
    min_length=1,
    max_length=255,
    pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$",
)
MEMORY_ID_PATH = Path(min_length=1, max_length=255)
LOGGER = logging.getLogger(__name__)
API_ERROR_RESPONSES = {
    code: {"model": ErrorView, "description": description}
    for code, description in {
        401: "Bearer authentication failed",
        502: "Model output validation failed",
        404: "Resource not found",
        409: "Operation conflicts with current state",
        413: "Request body too large",
        422: "Request validation failed",
        500: "Internal failure",
        503: "Required service unavailable",
    }.items()
}


def create_app(
    *,
    service: AgentAPIService | None = None,
    authenticator: BearerTokenAuthenticator | None = None,
    environment: Mapping[str, str] | None = None,
) -> FastAPI:
    selected_environment = os.environ if environment is None else environment
    selected_service = service or ProductionAgentAPIService(selected_environment)
    selected_authenticator = authenticator or BearerTokenAuthenticator.from_environment(
        selected_environment
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        selected_service.startup()
        try:
            yield
        finally:
            selected_service.shutdown()

    app = FastAPI(
        title="Japan Master Researcher API",
        version="1.0.0",
        description=(
            "Authenticated HTTP boundary for the durable LangGraph research workflow."
        ),
        lifespan=lifespan,
    )
    app.state.agent_service = selected_service
    app.state.authenticator = selected_authenticator

    origins = _cors_origins(selected_environment)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["Location", "X-Request-ID"],
        )
    app.add_middleware(RequestBodyLimitMiddleware)

    frontend_root = FilePath(__file__).resolve().parents[3] / "frontend"
    if frontend_root.is_dir():
        app.mount(
            "/frontend/assets",
            StaticFiles(directory=frontend_root),
            name="frontend-assets",
        )

        @app.get("/frontend", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(frontend_root / "index.html")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", "").strip()
        if not request_id or len(request_id) > 128:
            request_id = f"req_{uuid.uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.url.path == "/frontend" or request.url.path.startswith(
            "/frontend/assets/"
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ResourceNotFoundError)
    @app.exception_handler(NotFoundError)
    @app.exception_handler(OwnershipError)
    async def not_found_handler(_: Request, __: Exception) -> JSONResponse:
        return _error(404, "NOT_FOUND", "The requested resource was not found")

    @app.exception_handler(CaseBusyError)
    async def busy_handler(_: Request, __: CaseBusyError) -> JSONResponse:
        return _error(409, "CASE_BUSY", "Another operation is active for this case")

    @app.exception_handler(OperationConflictError)
    @app.exception_handler(IdempotencyConflictError)
    @app.exception_handler(VersionConflictError)
    @app.exception_handler(ConstraintError)
    async def conflict_handler(_: Request, exc: Exception) -> JSONResponse:
        detail = (
            str(exc)
            if isinstance(exc, OperationConflictError)
            else "The operation conflicts with the current resource state"
        )
        return _error(409, "OPERATION_CONFLICT", detail)

    @app.exception_handler(ServiceUnavailableError)
    @app.exception_handler(CheckpointerConfigurationError)
    @app.exception_handler(RepositoryConfigurationError)
    @app.exception_handler(StoreConfigurationError)
    @app.exception_handler(OperationalError)
    @app.exception_handler(InterfaceError)
    @app.exception_handler(UndefinedTable)
    async def unavailable_handler(_: Request, __: Exception) -> JSONResponse:
        return _error(503, "SERVICE_UNAVAILABLE", "A required service is unavailable")

    @app.exception_handler(APITimeoutError)
    @app.exception_handler(APIConnectionError)
    async def model_unavailable_handler(_: Request, __: Exception) -> JSONResponse:
        return _error(
            503,
            "MODEL_UNAVAILABLE",
            "Model service did not respond; refresh the case and retry "
            "its failed stage",
        )

    @app.exception_handler(ModelOutputValidationError)
    async def model_output_handler(
        request: Request, exc: ModelOutputValidationError
    ) -> JSONResponse:
        LOGGER.error(
            "Model output rejected path=%s node=%s reason=%s",
            request.url.path,
            exc.node_name,
            failure_reason_code(exc),
        )
        return _error(
            502,
            "MODEL_OUTPUT_INVALID",
            f"模型在 {exc.node_name} 阶段返回的数据格式不符合要求；"
            "请刷新 Case 并重试当前阶段",
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
        failure_id = f"failure_{uuid.uuid4().hex}"
        LOGGER.error(
            "HTTP failure id=%s path=%s type=%s reason=%s",
            failure_id,
            request.url.path,
            type(exc).__name__,
            failure_reason_code(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "The request could not be completed",
                    "failure_id": failure_id,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
        code = {
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            503: "SERVICE_UNAVAILABLE",
        }.get(exc.status_code, "HTTP_ERROR")
        message = str(exc.detail) if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message}},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = list(
            dict.fromkeys(
                ".".join(str(part) for part in error.get("loc", ()))
                for error in exc.errors()
            )
        )[:20]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "fields": fields,
                }
            },
        )

    Principal = Annotated[str, Depends(selected_authenticator)]

    @app.get("/health/live", tags=["health"])
    def live() -> Mapping[str, Any]:
        return liveness()

    @app.get("/health/ready", tags=["health"])
    def ready() -> JSONResponse:
        report = dict(selected_service.readiness())
        checks = dict(report.get("checks", {}))
        checks["authentication"] = (
            "ok" if selected_authenticator.configured else "failed"
        )
        report["checks"] = checks
        report["status"] = (
            "ok"
            if checks and all(value == "ok" for value in checks.values())
            else "failed"
        )
        code = 200 if report["status"] == "ok" else 503
        return JSONResponse(status_code=code, content=report)

    @app.post(
        "/api/v1/cases",
        response_model=CaseView,
        responses=API_ERROR_RESPONSES,
        status_code=status.HTTP_201_CREATED,
        tags=["cases"],
    )
    def create_case(
        body: CreateCaseRequest,
        principal: Principal,
        response: Response,
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                min_length=1,
                max_length=255,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$",
            ),
        ],
    ) -> Mapping[str, Any]:
        result = selected_service.start_case(
            user_id=principal,
            message=body.message,
            case_id=body.case_id,
            idempotency_key=idempotency_key,
        )
        response.headers["Location"] = f"/api/v1/cases/{result['case_id']}"
        return result

    @app.get(
        "/api/v1/session",
        tags=["session"],
        responses=API_ERROR_RESPONSES,
    )
    def session(principal: Principal) -> Mapping[str, str]:
        return {"user_id": principal}

    @app.get(
        "/api/v1/cases",
        response_model=CaseListView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def list_cases(principal: Principal) -> Mapping[str, Any]:
        return selected_service.list_cases(user_id=principal)

    @app.get(
        "/api/v1/cases/{case_id}/progress",
        response_model=CaseProgressView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def get_case_progress(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.get_progress(user_id=principal, case_id=case_id)

    @app.get(
        "/api/v1/cases/{case_id}/workspace",
        response_model=WorkspaceView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def get_workspace(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.get_workspace(user_id=principal, case_id=case_id)

    @app.get(
        "/api/v1/cases/{case_id}/conversation",
        response_model=ConversationView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def get_conversation(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.get_conversation(user_id=principal, case_id=case_id)

    @app.post(
        "/api/v1/cases/{case_id}/messages",
        response_model=CaseView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def send_message(
        case_id: Annotated[str, CASE_ID_PATH],
        body: SendMessageRequest,
        principal: Principal,
    ) -> Mapping[str, Any]:
        return selected_service.send_message(
            user_id=principal,
            case_id=case_id,
            message=body.message,
        )

    @app.post(
        "/api/v1/cases/{case_id}/resume",
        response_model=CaseView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def resume_case(
        case_id: Annotated[str, CASE_ID_PATH],
        body: ResumeCaseRequest,
        principal: Principal,
    ) -> Mapping[str, Any]:
        return selected_service.resume_case(
            user_id=principal,
            case_id=case_id,
            payload=body.payload,
            interrupt_token=body.interrupt_token,
        )

    @app.post(
        "/api/v1/cases/{case_id}/retry",
        response_model=CaseView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def retry_case(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.retry_case(user_id=principal, case_id=case_id)

    @app.get(
        "/api/v1/cases/{case_id}",
        response_model=CaseView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def get_case(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.get_case(user_id=principal, case_id=case_id)

    @app.get(
        "/api/v1/cases/{case_id}/result",
        response_model=FinalResultView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def get_result(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.get_result(user_id=principal, case_id=case_id)

    @app.get(
        "/api/v1/memories",
        response_model=MemoryListView,
        responses=API_ERROR_RESPONSES,
        tags=["memories"],
    )
    def list_memories(
        principal: Principal,
        kind: Annotated[str | None, Query(max_length=100)] = None,
        tags: Annotated[list[str] | None, Query()] = None,
        status_filter: Annotated[
            str, Query(alias="status", pattern="^(ACTIVE|SUPERSEDED|DELETED)$")
        ] = "ACTIVE",
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
    ) -> Mapping[str, Any]:
        return selected_service.call_memory(
            user_id=principal,
            tool_name="list_memories",
            arguments={
                "kind": kind,
                "tags": tags or [],
                "status": status_filter,
                "limit": limit,
            },
        )

    @app.get(
        "/api/v1/memories/{memory_id}",
        response_model=MemoryView,
        responses=API_ERROR_RESPONSES,
        tags=["memories"],
    )
    def get_memory(
        memory_id: Annotated[str, MEMORY_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        result = selected_service.call_memory(
            user_id=principal,
            tool_name="get_memory",
            arguments={"memory_id": memory_id},
        )
        if result.get("status") == "NO_RESULT":
            raise ResourceNotFoundError("memory not found")
        return result

    @app.post(
        "/api/v1/memories/upload",
        response_model=MemoryView,
        responses=API_ERROR_RESPONSES,
        status_code=status.HTTP_201_CREATED,
        tags=["memories"],
    )
    async def upload_memory(
        principal: Principal,
        file: UploadFile = File(...),
    ) -> Mapping[str, Any]:
        content = await file.read(MAX_FILE_BYTES + 1)
        try:
            return selected_service.upload_memory_document(
                user_id=principal,
                filename=file.filename or "",
                content=content,
            )
        except UploadValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/v1/memories/{memory_id}/download",
        responses=API_ERROR_RESPONSES,
        tags=["memories"],
    )
    def download_memory(
        memory_id: Annotated[str, MEMORY_ID_PATH], principal: Principal
    ) -> Response:
        from urllib.parse import quote

        filename, content = selected_service.get_memory_document(
            user_id=principal, memory_id=memory_id
        )
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{quote(filename)}"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post(
        "/api/v1/memories",
        response_model=MemoryView,
        responses=API_ERROR_RESPONSES,
        status_code=status.HTTP_201_CREATED,
        tags=["memories"],
    )
    def create_memory(
        body: CreateMemoryRequest, principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.call_memory(
            user_id=principal,
            tool_name="create_memory",
            arguments=body.model_dump(),
        )

    @app.patch(
        "/api/v1/memories/{memory_id}",
        response_model=MemoryView,
        responses=API_ERROR_RESPONSES,
        tags=["memories"],
    )
    def update_memory(
        memory_id: Annotated[str, MEMORY_ID_PATH],
        body: UpdateMemoryRequest,
        principal: Principal,
    ) -> Mapping[str, Any]:
        return selected_service.call_memory(
            user_id=principal,
            tool_name="update_memory",
            arguments={"memory_id": memory_id, **body.model_dump(exclude_none=True)},
        )

    @app.delete(
        "/api/v1/memories/{memory_id}",
        response_model=MemoryView,
        responses=API_ERROR_RESPONSES,
        tags=["memories"],
    )
    def delete_memory(
        memory_id: Annotated[str, MEMORY_ID_PATH],
        body: DeleteMemoryRequest,
        principal: Principal,
    ) -> Mapping[str, Any]:
        return selected_service.call_memory(
            user_id=principal,
            tool_name="delete_memory",
            arguments={"memory_id": memory_id, **body.model_dump()},
        )

    @app.get(
        "/api/v1/cases/{case_id}/deletion-plan",
        response_model=CaseDeletionView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def plan_deletion(
        case_id: Annotated[str, CASE_ID_PATH], principal: Principal
    ) -> Mapping[str, Any]:
        return selected_service.plan_case_deletion(user_id=principal, case_id=case_id)

    @app.delete(
        "/api/v1/cases/{case_id}",
        response_model=CaseDeletionView,
        responses=API_ERROR_RESPONSES,
        tags=["cases"],
    )
    def delete_case_route(
        case_id: Annotated[str, CASE_ID_PATH],
        body: ConfirmCaseDeletionRequest,
        principal: Principal,
    ) -> Mapping[str, Any]:
        return selected_service.confirm_case_deletion(
            user_id=principal, case_id=case_id, plan_token=body.plan_token
        )

    return app


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _cors_origins(environment: Mapping[str, str]) -> list[str]:
    raw = environment.get("JMR_CORS_ORIGINS_JSON", "").strip()
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [
        item
        for item in values
        if isinstance(item, str)
        and item.startswith(("http://", "https://"))
        and item != "*"
    ]


app = create_app()
