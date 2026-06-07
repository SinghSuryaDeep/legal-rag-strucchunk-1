"""
Hybrid Retriever with Reciprocal Rank Fusion
=============================================
Combines dense (BGE-M3) and sparse (BM25) retrieval using RRF.

RRF_score(d) = Σ_{r∈{dense,sparse}} 1 / (k + rank_r(d))

where k=60 is the standard smoothing constant (Cormack et al., 2009).
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever: dense embeddings + BM25, fused with RRF.

    Stores `embeddings`, `index`, `dimension` (dense) and `bm25`, `library`
    (sparse) as instance attributes so they can be cached and restored.
    `encoder` is NOT cached (not picklable); it is rebuilt from `model_name`
    on the first retrieve() call after a cache restore.
    """

    def __init__(
        self,
        chunks: List[str],
        model_name: str = "BAAI/bge-m3",
        rrf_k: int = 60,
    ):
        """
        Args:
            chunks:     List of text chunks to index.
            model_name: HuggingFace model for dense embeddings.
            rrf_k:      RRF smoothing constant (default 60).
        """
        self.chunks = chunks
        self.model_name = model_name
        self.rrf_k = rrf_k

        self._init_dense()
        self._init_sparse()
        logger.info(f"HybridRetriever ready: {len(chunks)} chunks")

    # ── Index construction ────────────────────────────────────────────────────

    def _init_dense(self):
        if not self.chunks:
            logger.warning("HybridRetriever: empty chunk list — dense retrieval disabled")
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

            logger.info("Encoding chunks for dense index...")
            self.embeddings = self.encoder.encode(
                self.chunks,
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=16,
            ).astype(np.float32)

            self.dimension = self.embeddings.shape[1]
            # Single-threaded FAISS (avoids macOS OpenMP segfault)
            faiss.omp_set_num_threads(1)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index: {self.index.ntotal} vectors")

        except ImportError as e:
            logger.warning(f"Dense retrieval unavailable ({e}). Using sparse-only.")
            self.encoder = None
            self.index = None
            self.embeddings = None
            self.dimension = 0

    def _init_sparse(self):
        try:
            import bm25s

            logger.info("Building BM25 index (bm25s)...")
            tokenized = bm25s.tokenize(self.chunks, stopwords="en")
            self.bm25 = bm25s.BM25()
            self.bm25.index(tokenized)
            self.library = "bm25s"

        except ImportError:
            try:
                from rank_bm25 import BM25Okapi

                logger.info("Building BM25 index (rank_bm25)...")
                self.bm25 = BM25Okapi([doc.lower().split() for doc in self.chunks])
                self.library = "rank_bm25"

            except ImportError:
                logger.error("No BM25 library available. Install bm25s or rank-bm25.")
                self.bm25 = None
                self.library = None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks using dense+sparse RRF.

        Returns:
            List of {"chunk_id": int, "text": str, "score": float}
        """
        dense_results = self._dense_retrieve(query, k=k * 2)
        sparse_results = self._sparse_retrieve(query, k=k * 2)
        fused = self._rrf_fusion(dense_results, sparse_results)

        top_k = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [
            {"chunk_id": idx, "text": self.chunks[idx], "score": score}
            for idx, score in top_k
        ]

    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        # encoder may be absent after cache restore — lazily rebuild
        encoder = getattr(self, "encoder", None)
        index = getattr(self, "index", None)

        if encoder is None and hasattr(self, "model_name"):
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name, device="cpu")
                encoder = self.encoder
            except Exception:
                pass

        if encoder is None or index is None:
            return []

        q_emb = encoder.encode([query], normalize_embeddings=True).astype(np.float32).flatten()
        # Numpy cosine similarity — avoids macOS FAISS/OpenMP segfault
        emb = getattr(self, "embeddings", None)
        if emb is None:
            return []
        sims = emb @ q_emb
        top_k_idx = np.argsort(sims)[-k:][::-1]
        return [(int(i), float(sims[i])) for i in top_k_idx]

    def _sparse_retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        bm25 = getattr(self, "bm25", None)
        if bm25 is None:
            return []

        library = getattr(self, "library", "bm25s")
        try:
            if library == "bm25s":
                import bm25s

                q_tok = bm25s.tokenize([query], stopwords="en")
                results, scores = bm25.retrieve(q_tok, k=k)
                return [(int(idx), float(s)) for idx, s in zip(results[0], scores[0])]
            else:
                scores = bm25.get_scores(query.lower().split())
                top_k = np.argsort(scores)[-k:][::-1]
                return [(int(i), float(scores[i])) for i in top_k]
        except Exception as e:
            logger.warning(f"Sparse retrieval failed: {e}")
            return []

    def _rrf_fusion(
        self,
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]],
    ) -> Dict[int, float]:
        rrf_k = getattr(self, "rrf_k", 60)
        rrf: Dict[int, float] = {}
        for rank, (doc_id, _) in enumerate(dense):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        return rrf
