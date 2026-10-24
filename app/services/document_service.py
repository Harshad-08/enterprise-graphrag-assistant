"""Document upload validation, storage, indexing, listing and deletion."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.ingestion.parsers import SUPPORTED_EXTENSIONS, DocumentParseError, detect_format
from app.models.schemas import DocumentRecord, IngestionResult
from app.services.container import Container

logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    pass


def sanitize_filename(filename: str) -> str:
    """Strip directories and unsafe characters; keep a safe extension."""
    name = Path(filename or "").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise UploadValidationError("Invalid filename.")
    stem, ext = Path(name).stem[:100], Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(f"Unsupported file type '{ext or 'none'}'. Allowed: PDF, DOCX, TXT.")
    return f"{stem}{ext}"


class DocumentService:
    def __init__(self, container: Container) -> None:
        self.c = container

    def upload(self, filename: str, content: bytes, overrides: dict[str, Any] | None = None) -> IngestionResult:
        safe_name = sanitize_filename(filename)
        if not content:
            raise UploadValidationError("Uploaded file is empty.")
        if len(content) > self.c.settings.max_upload_bytes:
            raise UploadValidationError(f"File exceeds the {self.c.settings.max_upload_mb} MB upload limit.")
        try:
            detect_format(safe_name, content)
        except DocumentParseError as exc:
            raise UploadValidationError(str(exc)) from exc
        result = self.c.ingestion.ingest_bytes(safe_name, content, overrides)
        upload_dir = self.c.settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"{result.document_id}__{safe_name}").write_bytes(content)
        return result

    def ingest_path(self, path: Path) -> IngestionResult:
        return self.c.ingestion.ingest_bytes(path.name, path.read_bytes())

    def list(self) -> list[DocumentRecord]:
        return self.c.store.list_documents()

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.c.store.get_document(document_id)

    def delete(self, document_id: str) -> bool:
        deleted = self.c.ingestion.delete(document_id)
        if deleted:
            for stored in self.c.settings.upload_dir.glob(f"{document_id}__*"):
                stored.unlink(missing_ok=True)
        return deleted

    def seed_directory(self, directory: Path) -> list[IngestionResult]:
        results = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(self.ingest_path(path))
        # Second pass: references to documents ingested later are linked now.
        self.c.ingestion.relink_documents()
        self.c.ingestion.refresh_conflict_resolutions()
        return results
