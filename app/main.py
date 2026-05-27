"""
FastAPI application entry point.

Wires routes, middleware, exception handlers, and startup hooks.
Run locally with:  uvicorn app.main:app --reload
"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import documents, questions
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.observability.tracing import configure_langsmith
from app.schemas import ErrorResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup / shutdown hooks."""
    configure_logging()
    configure_langsmith()
    logger = get_logger(__name__)
    logger.info("application_starting", extra={"settings": get_settings().model_dump(exclude={"groq_api_key", "anthropic_api_key", "openai_api_key", "langsmith_api_key"})})
    yield
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # --- Request ID + latency middleware ---
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = str(latency_ms)
        get_logger("app.access").info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    # --- Exception handlers ---
    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="invalid_request", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception):
        get_logger("app.errors").exception("unhandled_exception")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                detail="An unexpected error occurred.",
            ).model_dump(),
        )

    # --- Routes ---
    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(documents.router)
    app.include_router(questions.router)

    return app


app = create_app()
