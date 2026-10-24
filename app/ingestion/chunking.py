"""Section-aware chunking with stable chunk IDs, page numbers and section metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import Chunk, ParsedPage
from app.utils.text import short_hash, split_sentences

# Markdown headings ("# Access"), numbered headings ("2. Access Requirements",
# "3.1 Scope") — short lines without terminal punctuation.
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,80})$")


@dataclass
class _Block:
    text: str
    page: int | None
    section: str | None


def _is_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 90 and bool(_HEADING_RE.match(line)) and not line.endswith((".", ":", ";"))


def _sections(pages: list[ParsedPage]) -> list[_Block]:
    """Split pages into paragraph blocks, remembering the current section heading."""
    blocks: list[_Block] = []
    section: str | None = None
    for page in pages:
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                blocks.append(_Block("\n".join(buffer).strip(), page.page_number, section))
                buffer.clear()

        for line in page.text.splitlines():
            if _is_heading(line):
                flush()
                section = line.strip().lstrip("#").strip()
                continue
            if not line.strip():
                flush()
                continue
            buffer.append(line.rstrip())
        flush()
    return [b for b in blocks if b.text]


def chunk_document(
    document_id: str,
    pages: list[ParsedPage],
    *,
    max_chars: int = 700,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Greedy sentence packing within a section/page; a chunk never spans sections or pages.

    Keeping chunks inside one section and page means each citation can name an
    exact page/section instead of a range.
    """
    chunks: list[Chunk] = []
    groups: list[tuple[int | None, str | None, list[str]]] = []
    for block in _sections(pages):
        sentences = split_sentences(block.text)
        if groups and groups[-1][0] == block.page and groups[-1][1] == block.section:
            groups[-1][2].extend(sentences)
        else:
            groups.append((block.page, block.section, sentences))

    for page, section, sentences in groups:
        current: list[str] = []
        for sentence in sentences:
            candidate_len = sum(len(s) + 1 for s in current) + len(sentence)
            if current and candidate_len > max_chars:
                chunks.append(_make_chunk(document_id, len(chunks), current, page, section))
                current = current[-overlap_sentences:] if overlap_sentences else []
                # Avoid a chunk made only of the overlap sentence when the next one is huge.
                if current and sum(len(s) for s in current) + len(sentence) > max_chars:
                    current = []
            current.append(sentence)
        if current:
            chunks.append(_make_chunk(document_id, len(chunks), current, page, section))
    return chunks


def _make_chunk(document_id: str, index: int, sentences: list[str], page: int | None, section: str | None) -> Chunk:
    text = " ".join(sentences).strip()
    return Chunk(
        chunk_id=f"{document_id}::c{index:03d}",
        document_id=document_id,
        text=text,
        chunk_index=index,
        page=page,
        section=section,
        content_hash=short_hash(text, 16),
    )
