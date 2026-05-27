"""POST /documents — ingest document text."""
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.store import DocumentStore, get_store
from app.schemas import IngestDocumentRequest, IngestDocumentResponse
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=IngestDocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
)
def post_document(
    payload: IngestDocumentRequest,
    store: DocumentStore = Depends(get_store),
) -> IngestDocumentResponse:
    try:
        record = ingest_document(
            text=payload.text, store=store, metadata=payload.metadata
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return IngestDocumentResponse(
        doc_id=record.doc_id,
        chunk_count=record.chunk_count,
        char_count=record.char_count,
        created_at=record.created_at,
    )
