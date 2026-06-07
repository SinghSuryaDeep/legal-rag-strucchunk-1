"""
Recursive Character Text Splitter
==================================
Baseline chunking method that splits text recursively by separators,
then by character count. No structural awareness — standard approach
used by LangChain and similar frameworks.
"""

import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Recursive character text splitter (baseline method).

    Splits text by trying separators in order of preference.
    Falls back to character-level splitting as a last resort.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: float = 0.1,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size: Target tokens per chunk (approximated as chars/4).
            overlap:    Overlap ratio between consecutive chunks (0–1).
            separators: Ordered list of separators to try before char-split.
        """
        self.chunk_size = chunk_size * 4  # convert token budget to ~chars
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> List[Dict]:
        """
        Chunk plain text using recursive splitting.

        Args:
            text: Full document text.

        Returns:
            List of dicts: {"text": str, "chunk_index": int,
                            "char_count": int, "approx_tokens": int}
        """
        if not text or not text.strip():
            return []

        raw = self._recursive_split(text, self.separators)
        result = []
        for i, chunk_text in enumerate(raw):
            if chunk_text.strip():
                result.append({
                    "text": chunk_text,
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                    "approx_tokens": len(chunk_text) // 4,
                })

        logger.info(f"Created {len(result)} chunks using recursive splitting")
        return result

    # ── Internal split logic ──────────────────────────────────────────────────

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        for sep in separators:
            if sep == "":
                return self._split_by_char(text)
            if sep in text:
                splits = text.split(sep)
                return self._merge_splits(splits, sep)

        return self._split_by_char(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into target-size chunks with overlap."""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        overlap_size = int(self.chunk_size * self.overlap)

        for split in splits:
            split = split.strip()
            if not split:
                continue
            split_len = len(split) + len(separator)

            if current_len + split_len > self.chunk_size and current:
                chunk_text = separator.join(current)
                chunks.append(chunk_text)
                # Start new chunk with overlap from the end of the previous one
                overlap_text = chunk_text[-overlap_size:] if overlap_size > 0 else ""
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(split)
            current_len += split_len

        if current:
            chunks.append(separator.join(current))
        return chunks

    def _split_by_char(self, text: str) -> List[str]:
        """Last-resort: split at fixed character positions with overlap."""
        chunks: List[str] = []
        overlap_size = int(self.chunk_size * self.overlap)
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap_size
        return chunks
