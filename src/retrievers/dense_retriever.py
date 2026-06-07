"""
Dense Retriever
===============
Semantic vector search using sentence embeddings + FAISS.
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever using sentence embeddings and FAISS index."""

    def __init__(
        self,
        chunks: List[str],
        model_name: str = "BAAI/bge-m3",
    ):
        """
        Args:
            chunks:     List of text chunks to index.
            model_name: HuggingFace model name for SentenceTransformer.
        """
        self.chunks = chunks
        self.model_name = model_name
        self._build_index()

    def _build_index(self):
        """Encode all chunks and build a FAISS inner-product index."""
        if not self.chunks:
            logger.warning("DenseRetriever: empty chunk list — index not built")
            self.encoder = None
            self.index = None
            self.embeddings = None
            self.dimension = 0
            return

        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            # Force CPU: MPS exhausts shared memory when multiple retrievers
            # are built simultaneously on large corpora (97K+ chunks)
            logger.info(f"Loading embedding model: {self.model_name} (device=cpu)")
            self.encoder = SentenceTransformer(self.model_name, device="cpu")

            logger.info(f"Encoding {len(self.chunks)} chunks...")
            self.embeddings = self.encoder.encode(
                self.chunks,
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=16,
            ).astype(np.float32)

            self.dimension = self.embeddings.shape[1]
            # Use FAISS with single thread (avoids macOS OpenMP segfault)
            faiss.omp_set_num_threads(1)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)

            logger.info(f"FAISS index built: {self.index.ntotal} vectors")

        except ImportError as e:
            raise ImportError(
                f"Dense retrieval requires sentence-transformers and faiss-cpu: {e}"
            )

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks for a query.

        Returns:
            List of {"chunk_id": int, "text": str, "score": float}
        """
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            # Encoder was stripped during cache restore — reload from model_name
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
            encoder = self.encoder

        q_emb = encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32).flatten()

        # Numpy cosine similarity — avoids macOS FAISS/OpenMP segfault
        sims = self.embeddings @ q_emb          # (n_chunks,) inner products
        top_k_idx = np.argsort(sims)[-k:][::-1]

        return [
            {
                "chunk_id": int(idx),
                "text": self.chunks[idx],
                "score": float(sims[idx]),
            }
            for idx in top_k_idx
        ]
