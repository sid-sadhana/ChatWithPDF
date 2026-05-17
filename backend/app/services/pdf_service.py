"""PDF text extraction using pypdf."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.exceptions import BadRequestError

log = logging.getLogger(__name__)


@dataclass
class PDFContent:
    text_chunks: list[str] = field(default_factory=list)


class PDFService:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def extract(self, filepath: str) -> PDFContent:
        try:
            reader = PdfReader(filepath)
        except Exception as exc:
            raise BadRequestError(f"Cannot open PDF: {exc}") from exc

        full_text = ""
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"

        full_text = full_text.strip()
        if not full_text:
            raise BadRequestError("PDF appears to be empty or unreadable")

        full_text = self._clean(full_text)

        docs = self._splitter.create_documents([full_text])
        text_chunks = [d.page_content for d in docs if d.page_content.strip()]

        log.info("Extracted %d text chunks", len(text_chunks))
        return PDFContent(text_chunks=text_chunks)

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'[^\x00-\x7FÀ-ɏḀ-ỿ]', ' ', text)
        text = text.replace('\r\n', ' ').replace('\r', ' ')
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'[•●◦▪▶►◆★☆♦✓✗✔✘→←↑↓⇒⇐⇑⇓]', '', text)
        return text.strip()
