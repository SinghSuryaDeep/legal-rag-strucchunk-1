"""
Sparse Retriever (BM25)
=======================
Keyword-based retrieval using BM25. Supports bm25s (preferred) and rank_bm25.
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25-based sparse retriever."""

    def __init__(self, chunks: List[str]):
        """
        Args:
            chunks: List of text chunks to index.
        """
        self.chunks = chunks
        self._build_index()

    def _build_index(self):
        """Build BM25 index using whichever library is available."""
        try:
            import bm25s

            logger.info("Building BM25 index with bm25s...")
            tokenized = bm25s.tokenize(self.chunks, stopwords="en")
            self.bm25 = bm25s.BM25()
            self.bm25.index(tokenized)
            self.library = "bm25s"

        except ImportError:
            try:
                from rank_bm25 import BM25Okapi

                logger.info("Building BM25 index with rank_bm25...")
                tokenized = [doc.lower().split() for doc in self.chunks]
                self.bm25 = BM25Okapi(tokenized)
                self.library = "rank_bm25"

            except ImportError:
                raise ImportError(
                    "No BM25 library found. Install bm25s or rank-bm25."
                )

        logger.info(f"BM25 index built using {self.library}")

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.

        Returns:
            List of {"chunk_id": int, "text": str, "score": float}
        """
        library = getattr(self, "library", "bm25s")
        if library == "bm25s":
            import bm25s

            q_tokenized = bm25s.tokenize([query], stopwords="en")
            results, scores = self.bm25.retrieve(q_tokenized, k=k)
            return [
                {
                    "chunk_id": int(idx),
                    "text": self.chunks[idx],
                    "score": float(score),
                }
                for idx, score in zip(results[0], scores[0])
            ]
        else:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_k_idx = np.argsort(scores)[-k:][::-1]
            return [
                {
                    "chunk_id": int(idx),
                    "text": self.chunks[idx],
                    "score": float(scores[idx]),
                }
                for idx in top_k_idx
            ]
