"""POST /questions/{doc_id} — ask a question about a previously ingested document."""
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.store import DocumentStore, get_store
from app.observability.callbacks import TokenLatencyCallback
from app.observability.pricing import estimate_cost_usd
from app.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    QuestionResponseMetadata,
    SourceChunk,
)
from app.services.agent import answer_question

logger = get_logger(__name__)
router = APIRouter(prefix="/questions", tags=["questions"])


@router.post(
    "/{doc_id}",
    response_model=AskQuestionResponse,
    summary="Ask a question about a document",
)
def post_question(
    doc_id: str,
    payload: AskQuestionRequest,
    store: DocumentStore = Depends(get_store),
) -> AskQuestionResponse:
    record = store.get(doc_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    request_id = str(uuid.uuid4())
    callback = TokenLatencyCallback(request_id=request_id)
    start = time.perf_counter()

    try:
        result = answer_question(
            record=record,
            question=payload.question,
            callbacks=[callback],
        )
    except Exception as exc:
        logger.exception(
            "agent_invocation_failed",
            extra={"request_id": request_id, "doc_id": doc_id},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The agent failed to produce an answer. See server logs.",
        ) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    settings = get_settings()
    cost = estimate_cost_usd(
        model=settings.llm_model,
        input_tokens=callback.input_tokens,
        output_tokens=callback.output_tokens,
    )

    logger.info(
        "question_answered",
        extra={
            "request_id": request_id,
            "doc_id": doc_id,
            "latency_ms": latency_ms,
            "tool_calls": result["tool_calls"],
            "short_circuited": result["short_circuited"],
            "input_tokens": callback.input_tokens,
            "output_tokens": callback.output_tokens,
            "estimated_cost_usd": cost,
        },
    )

    return AskQuestionResponse(
        answer=result["answer"],
        source_chunks=[SourceChunk(**chunk) for chunk in result["source_chunks"]],
        metadata=QuestionResponseMetadata(
            doc_id=doc_id,
            latency_ms=latency_ms,
            input_tokens=callback.input_tokens or None,
            output_tokens=callback.output_tokens or None,
            total_tokens=callback.total_tokens or None,
            estimated_cost_usd=cost,
            tool_calls=result["tool_calls"],
            short_circuited=result["short_circuited"],
        ),
    )
