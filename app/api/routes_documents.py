"""Document management endpoints."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import container_dependency
from app.graph.store import GraphStoreError
from app.ingestion.parsers import DocumentParseError
from app.models.schemas import DocumentRecord, IngestionResult
from app.services.container import Container
from app.services.document_service import DocumentService, UploadValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IngestionResult, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    document_id: str | None = Form(None),
    doc_type: str | None = Form(None),
    department: str | None = Form(None),
    version: str | None = Form(None),
    status: str | None = Form(None),
    authority: str | None = Form(None),
    effective_date: date | None = Form(None),
    publication_date: date | None = Form(None),
    supersedes: str | None = Form(None, description="Semicolon-separated document IDs"),
    container: Container = Depends(container_dependency),
) -> IngestionResult:
    limit = container.settings.max_upload_bytes
    # Read at most limit+1 bytes so oversized uploads are rejected without buffering them fully.
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds the {container.settings.max_upload_mb} MB limit.")
    overrides = {
        "title": title, "document_id": document_id, "doc_type": doc_type, "department": department,
        "version": version, "status": status, "authority": authority, "effective_date": effective_date,
        "publication_date": publication_date, "supersedes": supersedes,
    }
    try:
        return DocumentService(container).upload(file.filename or "", content, overrides)
    except (UploadValidationError, DocumentParseError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentRecord])
def list_documents(container: Container = Depends(container_dependency)) -> list[DocumentRecord]:
    try:
        return DocumentService(container).list()
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{document_id}", response_model=DocumentRecord)
def get_document(document_id: str, container: Container = Depends(container_dependency)) -> DocumentRecord:
    try:
        record = DocumentService(container).get(document_id)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return record


@router.delete("/{document_id}")
def delete_document(document_id: str, container: Container = Depends(container_dependency)) -> dict:
    try:
        deleted = DocumentService(container).delete(document_id)
    except GraphStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"deleted": True, "document_id": document_id}
