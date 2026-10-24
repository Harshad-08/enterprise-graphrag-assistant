"""Document metadata extraction.

Documents may begin with a simple front-matter block::

    ---
    document_id: SEC-POL-001
    title: Information Security Policy
    effective_date: 2024-01-01
    supersedes: SEC-POL-001-V2
    ---

Values supplied explicitly at upload time override front matter. Missing or
invalid values are recorded as ``metadata_warnings`` instead of being guessed,
because conflict resolution must not rely on fabricated dates or authority.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from app.models.schemas import DocumentMetadata, DocumentStatus, ParsedPage
from app.utils.text import short_hash, slugify

_FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_LIST_FIELDS = {"supersedes", "governs", "applies_to", "depends_on", "related"}
_DATE_FIELDS = {"publication_date", "effective_date"}
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields, text[match.end():]


def strip_front_matter(pages: list[ParsedPage]) -> tuple[dict[str, str], list[ParsedPage]]:
    """Front matter can only appear at the start of the first page."""
    if not pages:
        return {}, pages
    fields, remainder = split_front_matter(pages[0].text)
    if not fields:
        return {}, pages
    first = ParsedPage(text=remainder.strip(), page_number=pages[0].page_number)
    rest = [first] + pages[1:] if first.text else pages[1:]
    return fields, rest


def _parse_date(value: str, field: str, warnings: list[str]) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        warnings.append(f"Invalid {field} '{value}' ignored (expected YYYY-MM-DD).")
        return None


def build_metadata(
    fields: dict[str, Any],
    *,
    filename: str,
    source_format: str,
    body_text: str,
    overrides: dict[str, Any] | None = None,
    known_authorities: set[str] | None = None,
) -> DocumentMetadata:
    merged: dict[str, Any] = dict(fields)
    for key, value in (overrides or {}).items():
        if value not in (None, "", []):
            merged[key] = value
    warnings: list[str] = []

    doc_id = str(merged.get("document_id", "")).strip()
    if not doc_id or not _DOC_ID_RE.match(doc_id):
        if doc_id:
            warnings.append(f"Invalid document_id '{doc_id}' replaced with a content hash.")
        # Stable ID: identical content uploaded twice maps to the same document.
        doc_id = f"DOC-{short_hash(slugify(Path(filename).stem) + body_text)}".upper()

    title = str(merged.get("title") or "").strip()
    if not title:
        first_line = next((ln.strip("# ").strip() for ln in body_text.splitlines() if ln.strip()), "")
        title = first_line[:120] or Path(filename).stem.replace("_", " ").title()

    data: dict[str, Any] = {
        "document_id": doc_id,
        "title": title,
        "doc_type": str(merged.get("doc_type") or "document").lower(),
        "department": merged.get("department") or None,
        "version": str(merged["version"]) if merged.get("version") else None,
        "filename": filename,
        "source_format": source_format,
    }

    for field in _DATE_FIELDS:
        raw = merged.get(field)
        if isinstance(raw, date):
            data[field] = raw
        elif raw:
            data[field] = _parse_date(str(raw), field, warnings)
        else:
            data[field] = None
            warnings.append(f"Missing {field}.")

    raw_status = str(merged.get("status") or "").lower().strip()
    try:
        data["status"] = DocumentStatus(raw_status) if raw_status else DocumentStatus.UNKNOWN
    except ValueError:
        warnings.append(f"Unknown status '{raw_status}' treated as unknown.")
        data["status"] = DocumentStatus.UNKNOWN
    if data["status"] == DocumentStatus.UNKNOWN:
        warnings.append("Document status unknown.")

    authority = str(merged.get("authority") or "").lower().strip() or None
    if authority and known_authorities is not None and authority not in known_authorities:
        warnings.append(f"Unknown authority '{authority}' cannot be used for precedence.")
        authority = None
    if authority is None:
        warnings.append("Source authority unknown.")
    data["authority"] = authority

    for field in _LIST_FIELDS:
        raw = merged.get(field)
        if isinstance(raw, list):
            values = [str(v).strip() for v in raw if str(v).strip()]
        elif raw:
            values = [v.strip() for v in str(raw).split(";") if v.strip()]
        else:
            values = []
        data[field] = values
    if doc_id in data["supersedes"]:
        warnings.append("Document cannot supersede itself; ignored.")
        data["supersedes"] = [d for d in data["supersedes"] if d != doc_id]

    data["metadata_warnings"] = warnings
    return DocumentMetadata(**data)
