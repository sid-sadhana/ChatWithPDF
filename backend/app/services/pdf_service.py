"""PDF text extraction with semantic chunking."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from app.services.exceptions import BadRequestError

log = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(
    r'\n(?='
    r'(?:CHAPTER|SECTION|ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|APPENDIX|ACKNOWLEDGMENT)'
    r'|(?:\d+\.[\d.]*\s+[A-Z])'      # numbered headings like "3.1 Methods"
    r'|(?:[A-Z][A-Z\s]{4,}(?:\n|$))'  # ALL CAPS lines (likely headings)
    r')',
    re.IGNORECASE,
)

PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


@dataclass
class PDFContent:
    text_chunks: list[str] = field(default_factory=list)


class PDFService:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._max_chunk = chunk_size
        self._overlap = chunk_overlap

    def extract(self, filepath: str) -> PDFContent:
        try:
            reader = PdfReader(filepath)
        except Exception as exc:
            raise BadRequestError(f"Cannot open PDF: {exc}") from exc

        pages_text: list[str] = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")

        full_text = "\n".join(pages_text).strip()
        if not full_text:
            raise BadRequestError("PDF appears to be empty or unreadable")

        full_text = self._clean(full_text)
        chunks = self._semantic_chunk(full_text)

        log.info("Extracted %d semantic chunks", len(chunks))
        return PDFContent(text_chunks=chunks)

    def _semantic_chunk(self, text: str) -> list[str]:
        sections = SECTION_PATTERN.split(text)
        sections = [s.strip() for s in sections if s.strip()]

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self._max_chunk:
                chunks.append(section)
            else:
                chunks.extend(self._split_section(section))

        return [c for c in chunks if len(c) >= 50]

    def _split_section(self, section: str) -> list[str]:
        paragraphs = PARAGRAPH_BREAK.split(section)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= self._max_chunk:
                current = f"{current} {para}".strip() if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > self._max_chunk:
                    chunks.extend(self._hard_split(para))
                else:
                    current = para
                    continue
                current = ""

        if current:
            chunks.append(current)

        if self._overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for word in words:
            if length + len(word) + 1 > self._max_chunk and current:
                chunks.append(" ".join(current))
                overlap_words = []
                overlap_len = 0
                for w in reversed(current):
                    if overlap_len + len(w) + 1 > self._overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1
                current = overlap_words
                length = overlap_len
            current.append(word)
            length += len(word) + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(prev_words):
                if overlap_len + len(w) + 1 > self._overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            prefix = " ".join(overlap_words)
            result.append(f"{prefix} {chunks[i]}".strip() if prefix else chunks[i])
        return result

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'[^\x00-\x7FÀ-ɏḀ-ỿ\n]', ' ', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[•●◦▪▶►◆★☆♦✓✗✔✘→←↑↓⇒⇐⇑⇓]', '', text)
        return text.strip()
