"""Small, dependency-free text helpers shared by BM25, assessment and verification."""

from __future__ import annotations

import hashlib
import re
import unicodedata

STOPWORDS = frozenset(
    """a an and are as at be been being by can could did do does doing for from had has have
    having how i if in into is it its itself me my of on or our ours should so than that the their
    them then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours about above after again against
    all am any because before below between both during each few further here just more most no nor
    not only other own same some such s t don now also per via within without whether company
    companys policy policies document documents please tell according applicable apply applies
    """.split()
)

# Negations matter for verification and contradiction detection, so they are kept
# out of the stopword list used there.
NEGATIONS = frozenset({"not", "no", "never", "prohibited", "forbidden", "cannot", "nor"})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stem(token: str) -> str:
    """Very light suffix stripping so 'devices'/'device' and 'requirements'/'required'/'requires'
    share a stem. Not a linguistic stemmer; it only needs to be consistent."""
    if token.isdigit() or len(token) <= 4:
        return token
    for suffix in ("ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            if suffix == "s" and token.endswith("ss"):
                continue
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, stemmed: bool = True) -> list[str]:
    text = re.sub(r"['\u2019]s\b", "", text.lower())  # possessives: "company's" -> "company"
    tokens = _TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemmed:
        tokens = [stem(t) for t in tokens]
    return tokens


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 1}


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*•]\s)|\n(?=\s*\d+[.)]\s)", text):
        block = " ".join(line.strip() for line in block.splitlines() if line.strip())
        block = re.sub(r"^[-*•]\s+", "", block)
        if not block:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(block) if s.strip())
    return sentences


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def numbers_in(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))
