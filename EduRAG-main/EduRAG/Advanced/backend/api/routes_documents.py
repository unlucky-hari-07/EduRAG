"""
backend/api/routes_documents.py

Document management endpoints:

    POST   /documents/upload
    GET    /documents
    DELETE /documents/{document_id}
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi import File as FastAPIFile

from backend import config
from backend.models.responses import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from backend.services.ingestion import (
    DocumentValidationError,
    delete_document,
    get_document,
    ingest_document,
    list_documents,
    validate_upload,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = FastAPIFile(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="Uploaded file has no filename.")

    raw_bytes = await file.read()

    try:
        validate_upload(file.filename, len(raw_bytes))
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    temp_name = f"{uuid.uuid4()}{Path(file.filename).suffix.lower()}"
    temp_path = config.UPLOAD_DIR / temp_name

    try:
        with open(temp_path, "wb") as f:
            f.write(raw_bytes)

        entry = ingest_document(temp_path, original_filename=file.filename)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never leak internal stack traces to the client
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the document.",
        ) from exc
    finally:
        # The extracted/embedded content is now in the index; the raw
        # uploaded file itself doesn't need to be kept on disk.
        if temp_path.exists():
            temp_path.unlink()

    return UploadResponse(**entry)


@router.get("", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        count=len(docs),
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str) -> DeleteResponse:
    if not document_id or not document_id.strip():
        raise HTTPException(status_code=422, detail="document_id must not be blank.")

    existing = get_document(document_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No document found with id '{document_id}'.")

    deleted = delete_document(document_id)
    return DeleteResponse(
        document_id=document_id,
        deleted=deleted,
        message="Document deleted successfully." if deleted else "Document could not be deleted.",
    )
