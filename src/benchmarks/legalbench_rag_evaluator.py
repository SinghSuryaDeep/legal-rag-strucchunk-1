"""
LegalBench-RAG Evaluator (Layer 1 — Independent Validation)
============================================================
Adapter to evaluate StrucChunk on the LegalBench-RAG benchmark.

CHANGES FROM v1:
  [BUG FIX]   v1 used passage-ID string matching as the relevance criterion.
              LegalBench-RAG ground truth is CHARACTER SPANS, not passage IDs.
              The correct evaluation checks whether a retrieved chunk
              CONTAINS or OVERLAPS the ground-truth character span.
              This is a fundamental change: span_overlap_recall() replaces
              passage_id_recall().

  [NEW]       Stratified sub-dataset sampling. The mini version has 4 sub-
              datasets (ContractNLI, CUAD, MAUD, PrivacyQA) with 194 queries
              each. When sampling n_queries < 776, we now sample uniformly
              across sub-datasets to maintain balance. v1 shuffled the full
              list, which could over-sample CUAD (the largest).

  [NEW]       Per-sub-dataset metric breakdown. Table 2 in the paper needs
              ContractNLI / CUAD / MAUD / PrivacyQA rows. v1 reported only
              overall metrics.

  [ENHANCED]  MAP@10 added to all metric outputs.

  [ENHANCED]  Embedding caching. Re-encoding the same chunks for each method
              wastes ~3× compute. Chunks are now keyed by a content hash;
              if a cache file exists, embeddings are loaded from disk.
              Reduces 4-hour runtime to ~1 hour for 776 queries.

  [CRITICAL NOTE] LegalBench-RAG evaluation paradigm differs from CrPC/CPC:
    CrPC/CPC  uses section-level matching: "does 'Section 167' appear in chunk?"
    LBR       uses character-span matching: "does retrieved text overlap span [X,Y]?"
    These are NOT comparable — explain the difference in paper Section 5.1.

Repository: https://github.com/zeroentropy-cc/legalbenchrag
Data format (from their JSON):
  {
    "query": "Consider the Non-Disclosure Agreement...",
    "snippets": [
      {"file_path": "contractnli/CopAcc_NDA.txt", "span": [11461, 11963], "answer": "..."}
    ]
  }
"""

import json
import re
import hashlib
import pickle
import logging
import random as _lbr_random   # module-level, matches official benchmark.py usage
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# LegalBench-RAG mini sub-datasets and their expected query counts
LBR_SUBDATASETS = {
    "contractnli": {"name": "ContractNLI", "expected_queries": 194},
    "cuad":        {"name": "CUAD",        "expected_queries": 194},
    "maud":        {"name": "MAUD",        "expected_queries": 194},
    "privacyqa":   {"name": "PrivacyQA",   "expected_queries": 194},
}


class LegalBenchRAGEvaluator:
    """
    Evaluates retrieval methods on LegalBench-RAG with proper char-span matching.

    The key evaluation method is span_overlap_recall(): it computes whether
    the retrieved chunk text contains or substantially overlaps the ground-truth
    character span from the source document.
    """

    def __init__(
        self,
        data_dir: str,
        n_queries: int = 776,
        random_seed: int = 42,
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 512,
        chunk_overlap: float = 0.15,
        # Paper-matching chunk size: 500 chars, no overlap (Section 4.2)
        # Set chunk_size_chars to override the token-based chunk_size.
        # Paper uses 500 chars fixed-size, no overlap for the Naive baseline;
        # RCTS is variable but targets similar sizes. Default None = use chunk_size.
        chunk_size_chars: int = 500,
        chunk_overlap_chars: float = 0.0,
        cache_dir: str = "cache/lbr_embeddings",
        min_overlap_ratio: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.n_queries = n_queries
        self.random_seed = random_seed
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Paper-matching: 500 chars, 0 overlap → token equivalent = 125 tokens
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.cache_dir = Path(cache_dir)
        self.min_overlap_ratio = min_overlap_ratio
        np.random.seed(random_seed)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"LegalBench-RAG data directory not found: {self.data_dir}\n"
                "Download from: https://github.com/zeroentropy-cc/legalbenchrag"
            )

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Load LegalBench-RAG queries and source documents.

        FIX: After sampling queries, documents are filtered to ONLY the files
        referenced by those queries. This matches the original paper's LBR-mini
        design where 776 queries map to exactly 72 documents (~4K chunks), not
        the full 714-document corpus (~97K chunks). Without this filter retrieval
        is penalised by 90K irrelevant chunks, causing PrivacyQA to score 0.000.

        Returns:
            queries: List of query dicts with id, query, snippets, subdataset
            documents: Dict[file_path → full_text]  ← filtered to query-referenced files only
        """
        raw_queries = self._load_raw_queries()

        # Sample queries FIRST, then load only the documents those queries need
        queries = self._stratified_sample(raw_queries, self.n_queries)
        logger.info(
            f"Sampled {len(queries)} queries "
            f"({', '.join(f'{k}: {v}' for k, v in self._count_by_subdataset(queries).items())})"
        )

        # Collect only the file paths referenced by the sampled queries
        referenced_files: set = set()
        for q in queries:
            for s in q.get("snippets", []):
                fp = s.get("file_path", "")
                if fp:
                    referenced_files.add(fp)
                    referenced_files.add(fp.split("/")[-1])   # stem fallback

        # Load only those documents (mini corpus)
        documents = self._load_documents(filter_files=referenced_files)
        logger.info(
            f"Mini corpus: {len(documents)//2} documents "
            f"(filtered from full corpus to match {len(referenced_files)//2} query-referenced files)"
        )
        return queries, documents

    def _load_raw_queries(self) -> List[Dict]:
        """Load the LegalBench-RAG JSON query file."""
        candidates = [
            self.data_dir / "tests.json",
            self.data_dir / "queries.json",
            self.data_dir / "data" / "tests.json",
        ]
        for path in candidates:
            if path.exists():
                logger.info(f"Loading queries from: {path}")
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                # Handle {"tests": [...]} wrapper or plain list
                if isinstance(data, dict) and "tests" in data:
                    queries = data["tests"]
                elif isinstance(data, list):
                    queries = data
                else:
                    queries = []

                # Normalize: ensure each query has id, query, snippets, subdataset
                normalized = []
                for i, q in enumerate(queries):
                    normalized.append({
                        "id": q.get("id", f"q_{i}"),
                        "query": q.get("query", q.get("question", "")),
                        "snippets": q.get("snippets", []),   # [{file_path, span, answer}]
                        "subdataset": self._infer_subdataset(q),
                    })
                return normalized

        # Standard LBR layout: data_dir/data/benchmarks/*.json (one file per sub-dataset)
        benchmarks_dir = self.data_dir / "data" / "benchmarks"
        if benchmarks_dir.exists():
            all_queries: List[Dict] = []
            for json_file in sorted(benchmarks_dir.glob("*.json")):
                # Normalise: privacy_qa.json → "privacyqa" to match LBR_SUBDATASETS key
                subdataset = json_file.stem.lower().replace("_", "")
                logger.info(f"Loading {subdataset} queries from: {json_file}")
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                tests = data.get("tests", data) if isinstance(data, dict) else data
                for i, q in enumerate(tests):
                    raw_snippets = q.get("snippets", [])
                    # LBR snippets: {"file_path": str, "span": [int, int]}
                    # span may be stored as list or tuple
                    normalized_snippets = [
                        {
                            "file_path": s.get("file_path", ""),
                            "span": list(s.get("span", [])),
                            "answer": s.get("answer", ""),
                        }
                        for s in raw_snippets
                    ]
                    all_queries.append({
                        "id": q.get("id", f"{subdataset}_{i}"),
                        "query": q.get("query", q.get("question", "")),
                        "snippets": normalized_snippets,
                        "subdataset": subdataset,
                    })
            if all_queries:
                logger.info(f"Loaded {len(all_queries)} queries from {benchmarks_dir}")
                return all_queries

        # Fallback: look for JSONL
        jsonl_files = list(self.data_dir.glob("**/*.jsonl"))
        if jsonl_files:
            with open(jsonl_files[0]) as f:
                queries = [json.loads(line) for line in f if line.strip()]
            return [
                {
                    "id": q.get("id", str(i)),
                    "query": q.get("query", q.get("question", "")),
                    "snippets": q.get("snippets", []),
                    "subdataset": self._infer_subdataset(q),
                }
                for i, q in enumerate(queries)
            ]

        raise FileNotFoundError(
            f"No LegalBench-RAG query file found in {self.data_dir}.\n"
            "The GitHub repo only contains code. Download the data from Dropbox:\n"
            "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/"
            "AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg\n"
            "Extract the zip so that data/benchmarks/ and data/corpus/ exist "
            f"inside {self.data_dir}/"
        )

    def _infer_subdataset(self, query: Dict) -> str:
        """Infer which sub-dataset a query belongs to from file_path hints."""
        snippets = query.get("snippets", [])
        if snippets:
            # Normalise: remove underscores so "privacy_qa/..." matches key "privacyqa"
            fp = snippets[0].get("file_path", "").lower().replace("_", "")
            for key in LBR_SUBDATASETS:
                if key in fp:
                    return key
        # Check query text hints
        q = query.get("query", "").lower()
        if "non-disclosure" in q or "nda" in q:
            return "contractnli"
        if "privacy" in q or "data practice" in q:
            return "privacyqa"
        if "merger" in q or "acquisition" in q or "m&a" in q:
            return "maud"
        return "cuad"  # default

    def _stratified_sample(self, queries: List[Dict], n: int) -> List[Dict]:
        """
        Select n queries with equal representation from each sub-dataset.

        FIX (corpus size): The original paper's LBR-mini was created by taking
        the FIRST 194 queries from each sub-dataset file, which maps to only
        ~60-72 documents (~6-8M chars, ~17K chunks). Random shuffling maps to
        all 714 documents (~80M chars, ~160K chunks), making retrieval 9× harder.

        Using ordered (first-n) selection closely reproduces the original paper's
        mini corpus size (our: 60 docs, 6.3M chars vs paper: 72 docs, 8.7M chars).
        """
        if n >= len(queries):
            return queries

        by_subdataset: Dict[str, List[Dict]] = {}
        for q in queries:
            sd = q.get("subdataset", "cuad")
            by_subdataset.setdefault(sd, []).append(q)

        n_per_dataset = n // len(by_subdataset)
        sampled = []

        for sd, qs in by_subdataset.items():
            # FIX: SORT_BY_DOCUMENT=True — exact copy of official benchmark.py lines 44-52.
            # Uses module-level _lbr_random (= random) to match official behaviour exactly.
            sorted_qs = sorted(
                qs,
                key=lambda q: (
                    _lbr_random.seed(q["snippets"][0]["file_path"] if q.get("snippets") else ""),
                    _lbr_random.random(),
                )[1],
            )
            sampled.extend(sorted_qs[:n_per_dataset])

        return sampled

    def _count_by_subdataset(self, queries: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for q in queries:
            sd = q.get("subdataset", "unknown")
            counts[sd] = counts.get(sd, 0) + 1
        return counts

    def _load_documents(self, filter_files: set = None) -> Dict[str, str]:
        """
        Load source documents, optionally filtered to a specific set of files.

        FIX (corpus size): When filter_files is provided, only loads documents
        whose relative path or stem appears in filter_files. This reduces the
        retrieval pool from ~97K chunks (full corpus) to ~4K chunks (mini corpus),
        matching the original LBR-mini evaluation setting from the paper.
        """
        docs: Dict[str, str] = {}

        corpus_dir = self.data_dir / "data" / "corpus"
        search_root = corpus_dir if corpus_dir.exists() else self.data_dir

        import unicodedata as _ud
        for txt_file in search_root.glob("**/*.txt"):
            rel_path = str(txt_file.relative_to(search_root))
            # Apply corpus filter if provided
            if filter_files is not None:
                # FIX: Normalise Unicode (NFC) — macOS stores filenames in NFD
                # (decomposed), but JSON references use NFC (precomposed).
                # e.g. "MOËT" on disk = M-O-E-̈-T (NFD) but in JSON = MOËT (NFC)
                rel_nfc = _ud.normalize("NFC", rel_path)
                stem_nfc = _ud.normalize("NFC", txt_file.stem)
                if rel_nfc not in filter_files and stem_nfc not in filter_files:
                    continue
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                docs[rel_path] = f.read()
            docs[txt_file.stem] = docs[rel_path]

        logger.info(f"Loaded {len(docs) // 2} source documents from {search_root}")
        return docs

    # ── Chunking ──────────────────────────────────────────────────────────────

    def build_chunk_pools(
        self,
        documents: Dict[str, str],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Build recursive and struc chunk pools from all source documents.

        Each chunk carries:
          text, doc_file_path, char_start, char_end  (for span matching)

        Returns (recursive_chunks, struc_chunks).
        """
        from src.chunkers.strucchunk import StrucChunker

        # FIX: Use paper-matching 500-char fixed-size chunks, no overlap.
        # RecursiveChunker uses token-based sizing which produces ~2000-char chunks.
        # For LBR we split directly at character boundaries to match the paper.
        struc_chunker = StrucChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            add_breadcrumbs=True,
            resolve_cross_refs=True,
        )

        recursive_chunks: List[Dict] = []
        struc_chunks: List[Dict] = []

        # Paper-matching: split at exactly chunk_size_chars with no overlap
        # Uses character-level fixed-size splitting, NOT token-based.
        cs = self.chunk_size_chars  # default 500 chars (paper Section 4.2)

        for file_path, text in documents.items():
            if not text.strip():
                continue
            # Skip stem-only keys (e.g. "Fiverr") — only process full-path keys
            # (e.g. "privacy_qa/Fiverr.txt") to avoid duplicate chunks
            if "/" not in file_path:
                continue

            # FIX: Exact 500-char fixed-size chunks, NO strip, NO overlap.
            # Official code (baseline.py line 93-95):
            #   for i in range(0, len(document.content), chunk_size):
            #       text_splits.append(document.content[i : i + chunk_size])
            # It asserts "".join(text_splits) == document.content — no stripping.
            # Stripping would shift char_start/char_end and break span matching.
            pos = 0
            while pos < len(text):
                chunk_text = text[pos:pos + cs]   # NO .strip() — exact positions
                if chunk_text:
                    recursive_chunks.append({
                        "text": chunk_text,
                        "doc_file_path": file_path,
                        "char_start": pos,
                        "char_end": pos + len(chunk_text),   # exact end
                    })
                pos += cs  # no overlap

            # StrucChunk (contract-adapted: uses section numbering if present)
            struc_chunks.extend(
                self._struc_chunk_contract(text, file_path, struc_chunker)
            )

        logger.info(
            f"Built chunk pools: {len(recursive_chunks)} recursive, "
            f"{len(struc_chunks)} strucchunk"
        )
        return recursive_chunks, struc_chunks

    def _struc_chunk_contract(
        self,
        text: str,
        file_path: str,
        struc_chunker,
    ) -> List[Dict]:
        """
        Apply StrucChunk structure detection to LegalBench-RAG contracts.

        Since LBR documents are contracts (not statutory codes), the hierarchical
        structure is typically: numbered sections (1, 1.1, 1.1.1) and defined
        terms. We detect this structure and apply breadcrumb + cross-ref
        augmentation accordingly.
        """
        section_pattern = re.compile(
            r'(?:^|\n)((?:\d+\.)+\d*\s+[A-Z][^\n]{5,80})', re.MULTILINE
        )
        cross_ref_pattern = re.compile(r'[Ss]ection\s+([\d\.]+)', re.IGNORECASE)

        # Find section boundaries in order
        section_matches = list(section_pattern.finditer(text))
        chunks = []

        if not section_matches:
            # Flat document — use recursive chunking with minimal breadcrumb
            from src.chunkers.recursive_chunker import RecursiveChunker
            rc = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            for chunk in rc.chunk(text):
                pos = text.find(chunk["text"][:80])
                chunks.append({
                    "text": chunk["text"],
                    "doc_file_path": file_path,
                    "char_start": pos if pos >= 0 else 0,
                    "char_end": pos + len(chunk["text"]) if pos >= 0 else len(chunk["text"]),
                    "has_structure": False,
                })
            return chunks

        # Chunk by section boundaries
        boundaries = [(m.start(), m.group(1).strip()) for m in section_matches]
        boundaries.append((len(text), "END"))

        for i in range(len(boundaries) - 1):
            sec_start, sec_title = boundaries[i]
            sec_end = boundaries[i + 1][0]
            sec_text = text[sec_start:sec_end].strip()

            if not sec_text:
                continue

            breadcrumb = f"[Contract > {sec_title[:60]}]"
            cross_refs = cross_ref_pattern.findall(sec_text)

            chunk_text = breadcrumb + "\n" + sec_text
            if cross_refs:
                unique_refs = list(dict.fromkeys(cross_refs))[:3]
                chunk_text += f"\n[References: Sections {', '.join(unique_refs)}]"

            # Split if too long
            approx_tokens = len(chunk_text) // 4
            if approx_tokens <= self.chunk_size * 1.5:
                chunks.append({
                    "text": chunk_text,
                    "doc_file_path": file_path,
                    "char_start": sec_start,   # [NEW]
                    "char_end": sec_end,       # [NEW]
                    "has_structure": True,
                })
            else:
                # Sub-split large sections
                mid = len(chunk_text) // 2
                for sub_text in [chunk_text[:mid], chunk_text[mid:]]:
                    sub_pos = text.find(sub_text[:80])
                    chunks.append({
                        "text": sub_text,
                        "doc_file_path": file_path,
                        "char_start": sub_pos if sub_pos >= 0 else sec_start,
                        "char_end": (sub_pos + len(sub_text)) if sub_pos >= 0 else sec_end,
                        "has_structure": True,
                    })

        return chunks

    # ── [NEW] Span-overlap relevance matching ─────────────────────────────────
    def compute_relevance_vector(
        self,
        retrieved_chunks: List[Dict],
        query: Dict,
        k: int = 10,
    ) -> List[int]:
        """
        [NEW — REPLACES passage_id_matching from v1]

        Compute binary relevance vector using CHARACTER-SPAN OVERLAP.

        LegalBench-RAG's ground truth is a list of (file_path, [start, end])
        spans. A retrieved chunk is RELEVANT if it substantially overlaps
        at least one ground-truth span.

        Overlap is measured as:
          overlap_ratio = len(chunk ∩ gt_span) / len(gt_span)

        A chunk is considered relevant if overlap_ratio ≥ min_overlap_ratio (0.5).

        This is fundamentally different from v1's passage-ID matching, which
        was a proxy metric that didn't match LBR's actual evaluation protocol.
        """
        snippets = query.get("snippets", [])
        if not snippets:
            return [0] * k

        # Build list of (file_path, start, end) tuples for ground truth
        gt_spans = []
        for snippet in snippets:
            fp = snippet.get("file_path", "")
            span = snippet.get("span", [])
            if fp and len(span) == 2:
                gt_spans.append((fp, int(span[0]), int(span[1])))

        if not gt_spans:
            return [0] * k

        relevance = []
        for chunk in retrieved_chunks[:k]:
            chunk_fp = chunk.get("doc_file_path", "")
            chunk_start = chunk.get("char_start", 0)
            chunk_end = chunk.get("char_end", chunk_start + len(chunk.get("text", "")))

            is_relevant = False
            for gt_fp, gt_start, gt_end in gt_spans:
                # File path match (flexible: stem or relative path)
                fp_match = (
                    chunk_fp == gt_fp
                    or Path(chunk_fp).stem == Path(gt_fp).stem
                    or gt_fp in chunk_fp
                    or chunk_fp in gt_fp
                )
                if not fp_match:
                    # Try text-based fallback: check if GT answer text appears in chunk
                    for snippet in snippets:
                        answer = snippet.get("answer", "")
                        if answer and len(answer) > 30:
                            if answer[:100] in chunk.get("text", ""):
                                is_relevant = True
                                break
                    if is_relevant:
                        break
                    continue

                # Compute span overlap
                overlap_start = max(chunk_start, gt_start)
                overlap_end = min(chunk_end, gt_end)
                overlap_len = max(0, overlap_end - overlap_start)
                gt_len = gt_end - gt_start

                if gt_len > 0 and (overlap_len / gt_len) >= self.min_overlap_ratio:
                    is_relevant = True
                    break

            relevance.append(1 if is_relevant else 0)

        while len(relevance) < k:
            relevance.append(0)

        return relevance[:k]
    # ── [END NEW] ─────────────────────────────────────────────────────────────

    # ── Retriever building with caching ──────────────────────────────────────

    def _build_retriever_with_cache(
        self,
        chunks: List[Dict],
        method_name: str,
        retriever_class,
        **retriever_kwargs,
    ):
        """
        [NEW] Build retriever with embedding cache.

        Embeddings for the same set of chunks are computed once and cached to
        disk. Subsequent runs load from cache, reducing 4-hour runtime to ~1h.
        """
        # Create cache key from chunk texts hash
        texts = [c["text"] for c in chunks]
        content_hash = hashlib.md5(json.dumps(texts[:100]).encode()).hexdigest()[:8]
        cache_path = self.cache_dir / f"{method_name}_{content_hash}.pkl"

        if cache_path.exists():
            logger.info(f"  Loading cached embeddings for {method_name}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            retriever = retriever_class.__new__(retriever_class)
            retriever.__dict__.update(cached_data)
            retriever.chunks = texts  # restore reference
            # Inject kwargs attrs not saved in old caches (e.g. model_name, rrf_k)
            for k, v in retriever_kwargs.items():
                if not hasattr(retriever, k):
                    setattr(retriever, k, v)
            # encoder is not picklable — rebuild from model_name
            if hasattr(retriever, "model_name") and not hasattr(retriever, "encoder"):
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"  Restoring encoder ({retriever.model_name})...")
                    retriever.encoder = SentenceTransformer(retriever.model_name)
                except Exception as e:
                    logger.warning(f"  Could not restore encoder: {e}")
            return retriever

        logger.info(f"  Building {method_name} retriever (will cache)...")
        retriever = retriever_class(chunks=texts, **retriever_kwargs)

        # Cache embeddings if available
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cacheable = {}
        for attr in ["embeddings", "index", "dimension", "bm25"]:
            if hasattr(retriever, attr):
                cacheable[attr] = getattr(retriever, attr)
        if cacheable:
            with open(cache_path, "wb") as f:
                pickle.dump(cacheable, f)
            logger.info(f"  Cached {method_name} embeddings → {cache_path}")

        return retriever

    # ── Main evaluation pipeline ─────────────────────────────────────────────

    def run_evaluation(self, output_dir: str = "results/legalbench_rag") -> Dict:
        """
        Full LegalBench-RAG evaluation pipeline.

        IMPORTANT: Uses character-span matching, not passage-ID matching.
        See compute_relevance_vector() for details.

        Returns metrics broken down by sub-dataset and overall.
        """
        from src.retrievers.dense_retriever import DenseRetriever
        from src.retrievers.sparse_retriever import SparseRetriever
        from src.retrievers.hybrid_retriever import HybridRetriever
        from src.evaluation.metrics import RetrievalMetrics
        from src.evaluation.statistical_tests import StatisticalTests

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        queries, documents = self.load_data()

        # Build chunk pools
        logger.info("Building chunk pools from LBR documents...")
        recursive_chunks, struc_chunks = self.build_chunk_pools(documents)

        # Build retrievers (with embedding cache)
        logger.info("Building retrievers...")
        methods = {
            "Recursive + Dense": self._build_retriever_with_cache(
                recursive_chunks, "rec_dense",
                DenseRetriever, model_name=self.embedding_model
            ),
            "BM25 Only": self._build_retriever_with_cache(
                recursive_chunks, "bm25",
                SparseRetriever
            ),
            "Naive Hybrid": self._build_retriever_with_cache(
                recursive_chunks, "naive_hybrid",
                HybridRetriever, model_name=self.embedding_model
            ),
            "StrucChunk + Hybrid": self._build_retriever_with_cache(
                struc_chunks, "struc_hybrid",
                HybridRetriever, model_name=self.embedding_model
            ),
        }
        chunk_maps = {
            name: (struc_chunks if "StrucChunk" in name else recursive_chunks)
            for name in methods
        }

        # FIX: evaluate at the same k values as the original paper (Table 4/5)
        # Paper reports: k = 1, 2, 4, 8, 16, 32, 64
        # We also keep k=10 for comparison with CrPC/GDPR experiments
        k_values = [1, 2, 4, 8, 10, 16, 32, 64]
        k_max = max(k_values)
        all_results: Dict[str, List[Dict]] = {}

        for method_name, retriever in methods.items():
            logger.info(f"Evaluating: {method_name} ({len(queries)} queries)...")
            method_results = []

            for query in queries:
                # Retrieve up to k_max chunks once; slice for each k
                retrieved = retriever.retrieve(query["query"], k=k_max)
                chunks_for_method = chunk_maps[method_name]
                retrieved_chunks = [
                    chunks_for_method[r["chunk_id"]]
                    if r["chunk_id"] < len(chunks_for_method) else {"text": ""}
                    for r in retrieved
                ]

                # Binary relevance vector (k_max length) for MRR/MAP/nDCG
                rel_vector = self.compute_relevance_vector(
                    retrieved_chunks, query, k=k_max
                )

                # FIX: character-level Recall@k and Precision@k (original paper metric)
                char_recall = {}
                char_precision = {}
                for k in k_values:
                    r, p = self._char_recall_precision_at_k(
                        retrieved_chunks[:k], query
                    )
                    char_recall[k] = r
                    char_precision[k] = p

                method_results.append({
                    "query_id": query["id"],
                    "subdataset": query.get("subdataset", "unknown"),
                    "relevance": rel_vector,
                    "char_recall": char_recall,
                    "char_precision": char_precision,
                })

            all_results[method_name] = method_results

        # Compute metrics (binary + char-level)
        metrics_calc = RetrievalMetrics()
        metrics = self._compute_metrics(all_results, metrics_calc, k_values)

        # Statistical tests (use char-recall@8 to match paper's primary metric)
        stat_calc = StatisticalTests()
        stat_tests = self._run_stat_tests(all_results, stat_calc)

        # Save and print
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(output_path / f"lbr_metrics_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(output_path / f"lbr_stats_{timestamp}.json", "w") as f:
            json.dump(stat_tests, f, indent=2)

        self._print_results(metrics, stat_tests)

        return {"metrics": metrics, "statistical_tests": stat_tests}

    # ── Character-level metrics (original paper's metric) ────────────────────

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping/adjacent spans. Input must be sorted by start."""
        if not spans:
            return []
        merged = [list(spans[0])]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(a, b) for a, b in merged]

    def _char_recall_precision_at_k(
        self,
        retrieved_chunks: List[Dict],
        query: Dict,
    ) -> Tuple[float, float]:
        """
        Character-level Recall and Precision (original LBR paper metric).

        Paper definition (Section 4, Tables 4/5):
          Recall@k  = chars from GT covered by union of top-k chunks / total GT chars
          Precision@k = chars correctly retrieved / total chars retrieved

        This is the metric used in the original paper, NOT the binary 50% overlap.
        Both metrics avoid double-counting via span merging.
        """
        snippets = query.get("snippets", [])
        if not snippets:
            return 0.0, 0.0

        # Build ground truth per file (merged spans)
        gt_by_file: Dict[str, List[Tuple[int, int]]] = {}
        for s in snippets:
            fp = s.get("file_path", "")
            span = s.get("span", [])
            if fp and len(span) == 2:
                gt_by_file.setdefault(fp, []).append((int(span[0]), int(span[1])))

        merged_gt: Dict[str, List[Tuple[int, int]]] = {}
        total_gt_chars = 0
        for fp, spans in gt_by_file.items():
            m = self._merge_spans(sorted(spans))
            merged_gt[fp] = m
            total_gt_chars += sum(e - s for s, e in m)

        if total_gt_chars == 0 or not retrieved_chunks:
            return 0.0, 0.0

        # Collect retrieved spans per file
        ret_by_file: Dict[str, List[Tuple[int, int]]] = {}
        total_ret_chars = 0
        for chunk in retrieved_chunks:
            fp = chunk.get("doc_file_path", "")
            cs = chunk.get("char_start", 0)
            ce = chunk.get("char_end", cs + len(chunk.get("text", "")))
            if fp and ce > cs:
                ret_by_file.setdefault(fp, []).append((cs, ce))
                # also index by stem for flexible matching
                stem = Path(fp).stem
                ret_by_file.setdefault(stem, []).append((cs, ce))
            total_ret_chars += max(0, ce - cs)

        # Compute intersection per GT file
        total_overlap = 0
        for fp, gt_spans in merged_gt.items():
            # Try exact path, then stem
            ret_spans_raw = ret_by_file.get(fp) or ret_by_file.get(Path(fp).stem, [])
            if not ret_spans_raw:
                continue
            ret_merged = self._merge_spans(sorted(ret_spans_raw))
            for g_s, g_e in gt_spans:
                for r_s, r_e in ret_merged:
                    total_overlap += max(0, min(g_e, r_e) - max(g_s, r_s))

        recall = total_overlap / total_gt_chars
        precision = total_overlap / max(total_ret_chars, 1)
        return float(recall), float(precision)

    # ── Metrics computation ───────────────────────────────────────────────────

    def _compute_metrics(self, all_results: Dict, metrics_calc, k_values: List[int] = None) -> Dict:
        """
        Compute metrics overall AND per sub-dataset.

        FIX: Adds character-level Recall@k and Precision@k at all paper k values.
        FIX: Equal sub-dataset weighting for the "ALL" row matches the paper:
             "we weight the metrics equally on each dataset, independently from
              the number of documents or queries they contain." (Section 4.2)
        """
        if k_values is None:
            k_values = [1, 2, 4, 8, 10, 16, 32, 64]

        computed = {}

        for method_name, results in all_results.items():
            rel_all = np.array([r["relevance"] for r in results])

            def _char_avg(key, k):
                vals = [r["char_recall" if key == "recall" else "char_precision"].get(k, 0.0)
                        for r in results]
                return float(np.mean(vals)) if vals else 0.0

            overall_binary = {
                "recall@1":  float(np.mean(rel_all[:, 0])),
                "recall@5":  float(np.mean(np.max(rel_all[:, :5], axis=1))),
                "recall@10": float(np.mean(np.max(rel_all[:, :10], axis=1))),
                "mrr":       metrics_calc.mrr(rel_all),
                "ndcg@10":   metrics_calc.ndcg(rel_all, k=10),
                "map@10":    metrics_calc.map_at_k(rel_all, k=10),
                "n_queries": len(results),
            }

            # Character-level metrics at each k (paper metric)
            char_metrics = {}
            for k in k_values:
                char_metrics[f"char_recall@{k}"]    = _char_avg("recall", k)
                char_metrics[f"char_precision@{k}"] = _char_avg("precision", k)
            overall_binary.update(char_metrics)

            # Per-sub-dataset breakdown
            by_subdataset: Dict[str, Dict] = {}
            sd_char_recalls: Dict[str, Dict[int, float]] = {}  # for equal-weight overall

            for sd_key, sd_info in LBR_SUBDATASETS.items():
                sd_results = [r for r in results if r.get("subdataset") == sd_key]
                if not sd_results:
                    continue
                sd_rel = np.array([r["relevance"] for r in sd_results])

                def _sd_char_avg(key, k):
                    vals = [r["char_recall" if key == "recall" else "char_precision"].get(k, 0.0)
                            for r in sd_results]
                    return float(np.mean(vals)) if vals else 0.0

                sd_entry = {
                    "recall@1":  float(np.mean(sd_rel[:, 0])),
                    "recall@5":  float(np.mean(np.max(sd_rel[:, :5], axis=1))),
                    "recall@10": float(np.mean(np.max(sd_rel[:, :10], axis=1))),
                    "mrr":       metrics_calc.mrr(sd_rel),
                    "ndcg@10":   metrics_calc.ndcg(sd_rel, k=10),
                    "map@10":    metrics_calc.map_at_k(sd_rel, k=10),
                    "n_queries": len(sd_results),
                }
                sd_char_recalls[sd_info["name"]] = {}
                for k in k_values:
                    cr = _sd_char_avg("recall", k)
                    cp = _sd_char_avg("precision", k)
                    sd_entry[f"char_recall@{k}"]    = cr
                    sd_entry[f"char_precision@{k}"] = cp
                    sd_char_recalls[sd_info["name"]][k] = cr

                by_subdataset[sd_info["name"]] = sd_entry

            # FIX: Equal-weight overall char metrics (paper's "ALL" row formula)
            if sd_char_recalls:
                equal_weight = {}
                for k in k_values:
                    vals = [sd_char_recalls[sd][k] for sd in sd_char_recalls if k in sd_char_recalls[sd]]
                    equal_weight[f"char_recall@{k}_equalwt"] = float(np.mean(vals)) if vals else 0.0
                overall_binary.update(equal_weight)

            computed[method_name] = {"overall": overall_binary, "by_subdataset": by_subdataset}

        return computed

    def _run_stat_tests(self, all_results: Dict, stat_calc) -> Dict:
        our_method = "StrucChunk + Hybrid"
        if our_method not in all_results:
            return {}

        our_scores = [max(r["relevance"][:5]) for r in all_results[our_method]]
        tests = {}

        for baseline in ["Recursive + Dense", "BM25 Only", "Naive Hybrid"]:
            if baseline not in all_results:
                continue
            baseline_scores = [max(r["relevance"][:5]) for r in all_results[baseline]]
            blocks, _ = stat_calc.run_full_significance_table(
                our_scores,
                {f"vs {baseline}": baseline_scores},
                dataset_label="LBR-mini",
            )
            tests[f"{our_method} vs {baseline}"] = blocks[0]

        return tests

    def _print_results(self, metrics: Dict, stat_tests: Dict):
        """
        Print results matching the original LBR paper's table format (Tables 4/5).

        Shows:
          1. Paper-style Char Precision@k / Recall@k table (k=1,2,4,8,16,32,64)
          2. Per-sub-dataset char-recall (paper's equal-weighted ALL row)
          3. Binary Recall@k for CrPC/GDPR comparability
          4. Statistical significance
        """
        k_paper = [1, 2, 4, 8, 16, 32, 64]

        print("\n" + "=" * 110)
        print("LEGALBENCH-RAG RESULTS — Character-Level Evaluation")
        print("(Matching original paper Tables 4/5: Char Recall@k and Precision@k)")
        print("=" * 110)

        # ── Table A: Char Recall@k (paper format) ───────────────────────────
        hdr = f"{'Method':<28}" + "".join(f"  R@{k:>2}" for k in k_paper)
        print(f"\nChar Recall@k (↑ higher is better)\n{hdr}")
        print("-" * 95)
        for method, m in metrics.items():
            o = m["overall"]
            row = f"{method:<28}" + "".join(
                f"  {o.get(f'char_recall@{k}', 0)*100:>5.2f}" for k in k_paper
            )
            print(row)
        print("-" * 95)

        # ── Table B: Char Precision@k (paper format) ────────────────────────
        hdr2 = f"{'Method':<28}" + "".join(f"  P@{k:>2}" for k in k_paper)
        print(f"\nChar Precision@k (↑ higher is better)\n{hdr2}")
        print("-" * 95)
        for method, m in metrics.items():
            o = m["overall"]
            row = f"{method:<28}" + "".join(
                f"  {o.get(f'char_precision@{k}', 0)*100:>5.2f}" for k in k_paper
            )
            print(row)
        print("-" * 95)

        # ── Table C: Per-sub-dataset char-recall@8 (equal weighted, paper's ALL formula) ──
        print(f"\nChar Recall@8 per sub-dataset (equal-weighted, matches paper's ALL row)")
        print(f"{'Method':<28} {'ContractNLI':>13} {'CUAD':>8} {'MAUD':>8} {'PrivacyQA':>11} {'ALL(eq-wt)':>12}")
        print("-" * 85)
        for method, m in metrics.items():
            bsd = m.get("by_subdataset", {})
            cnli = bsd.get("ContractNLI", {}).get("char_recall@8", 0.0) * 100
            cuad = bsd.get("CUAD",        {}).get("char_recall@8", 0.0) * 100
            maud = bsd.get("MAUD",        {}).get("char_recall@8", 0.0) * 100
            pqa  = bsd.get("PrivacyQA",   {}).get("char_recall@8", 0.0) * 100
            all_ew = m["overall"].get("char_recall@8_equalwt", 0.0) * 100
            print(f"{method:<28} {cnli:>13.2f} {cuad:>8.2f} {maud:>8.2f} {pqa:>11.2f} {all_ew:>12.2f}")
        print("-" * 85)

        # ── Table D: Paper baseline comparison ──────────────────────────────
        print("\n--- COMPARISON WITH ORIGINAL PAPER (Tables 4/5) ---")
        print("Original paper best (RCTS, no reranker):  P@1=14.38  R@64=84.19 (PrivacyQA)")
        print("                                          P@1= 6.63  R@64=61.72 (ContractNLI)")
        print("                                          P@1= 2.65  R@64=28.28 (MAUD)")
        print("                                          P@1= 1.97  R@64=74.70 (CUAD)")
        if "Recursive + Dense" in metrics:
            o = metrics["Recursive + Dense"]["overall"]
            bsd = metrics["Recursive + Dense"].get("by_subdataset", {})
            print(f"Your Recursive+Dense:                     "
                  f"P@1={o.get('char_precision@1',0)*100:>5.2f}  "
                  f"R@64={o.get('char_recall@64',0)*100:>5.2f} (overall)")

        # ── Table E: Binary recall (for CrPC/GDPR comparability) ───────────
        print(f"\nBinary Recall@k (for comparison with CrPC/GDPR — NOT paper metric)")
        hdr3 = f"{'Method':<28} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'nDCG@10':>9}"
        print(hdr3)
        print("-" * 70)
        for method, m in metrics.items():
            o = m["overall"]
            print(f"{method:<28} {o['recall@1']:>7.3f} {o['recall@5']:>7.3f} "
                  f"{o['recall@10']:>7.3f} {o['mrr']:>7.3f} {o['ndcg@10']:>9.3f}")
        print("-" * 70)

        # ── Statistical significance ─────────────────────────────────────────
        if stat_tests:
            print("\nStatistical tests (Wilcoxon + BH correction):")
            cohens_d_hdr = "Cohen's d"
            print(f"{'Comparison':<40} {'p (BH-corr)':>12} {cohens_d_hdr:>11} {'95% CI':>16} {'Sig':>5}")
            print("-" * 86)
            for comp, result in stat_tests.items():
                print(
                    f"{comp:<40} "
                    f"{result.get('corrected_p_display', 'N/A'):>12} "
                    f"{result.get('cohens_d', 0.0):>11.3f} "
                    f"{result.get('ci_display', 'N/A'):>16} "
                    f"{result.get('corrected_sig_display', 'N/A'):>5}"
                )
            print("* α=0.05 after Benjamini-Hochberg FDR correction\n")
