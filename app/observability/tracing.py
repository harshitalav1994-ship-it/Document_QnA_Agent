"""
LangSmith tracing bootstrap.

If LANGSMITH_API_KEY is set and LANGSMITH_TRACING is true, we enable
LangSmith automatically. LangChain reads these env vars itself, so we
just have to set them. The fallback when LangSmith is not configured
is the callback-based logging in callbacks.py.
"""
import os

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_langsmith() -> bool:
    """Returns True if LangSmith tracing was enabled."""
    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        logger.info("langsmith_tracing_disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info(
        "langsmith_tracing_enabled",
        extra={"project": settings.langsmith_project},
    )
    return True
