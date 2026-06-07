"""
Retrieval Evaluation Metrics
============================
Standard IR metrics for evaluating retrieval performance.

CHANGES FROM v1:
  [NEW METRIC]  map_at_k()  — MAP@10, the standard missing from v1.
                For queries with exactly one relevant item, MAP@k = MRR@k.
                For cross-reference queries with 2+ relevant items,
                MAP@k correctly averages precision at each relevant rank,
                while MRR only captures the first relevant hit.
  [NEW METHOD]  per_query_scores() — returns per-query R@5, MRR, MAP for
                diagnostics and per-difficulty breakdowns.
  [ENHANCED]    compute_all() now includes MAP@10 and returns a flat dict
                suitable for direct insertion into LaTeX tables.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Standard retrieval evaluation metrics.

    Metrics covered:
      Recall@k    — primary metric for legal retrieval (section-level)
      Precision@k — complementary to recall
      MRR         — mean reciprocal rank (best for single-relevant queries)
      nDCG@k      — position-weighted relevance, comparable to BEIR papers
      MAP@k       — [NEW] mean average precision; handles multi-relevant queries
    """

    def recall_at_k(self, relevance: np.ndarray, k: int) -> float:
        """Recall@k: fraction of queries where ≥1 relevant item in top-k."""
        k = min(k, relevance.shape[1])
        return float(np.mean(np.max(relevance[:, :k], axis=1)))

    def precision_at_k(self, relevance: np.ndarray, k: int) -> float:
        """Precision@k: fraction of top-k items that are relevant."""
        k = min(k, relevance.shape[1])
        return float(np.mean(np.sum(relevance[:, :k], axis=1) / k))

    def mrr(self, relevance: np.ndarray) -> float:
        """
        Mean Reciprocal Rank.
        For tasks with a single relevant item per query, this equals MAP.
        For cross-reference queries with multiple relevant items, use MAP instead.
        """
        scores = []
        for row in relevance:
            hits = np.where(row == 1)[0]
            scores.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)
        return float(np.mean(scores))

    # ── [NEW METRIC] ─────────────────────────────────────────────────────────
    def map_at_k(self, relevance: np.ndarray, k: int = 10) -> float:
        """
        Mean Average Precision at k.

        WHY THIS MATTERS: MRR only captures the rank of the *first* relevant
        item. For cross-reference queries where expected_sections has 2+
        entries, MAP@k properly rewards retrieving all relevant items high in
        the ranking. A query expecting sections [167, 57] that gets rank-1=167,
        rank-2=57 scores MAP@10=1.0; a query that gets rank-1=167, rank-8=57
        scores MAP@10=0.56. MRR would score both as 1.0.

        For queries with exactly ONE relevant item, MAP@k == MRR@k.

        Standard in TREC, BEIR, and SIGIR evaluation papers.
        """
        k = min(k, relevance.shape[1])
        scores = []
        for row in relevance:
            hits = np.where(row[:k] == 1)[0]
            if len(hits) == 0:
                scores.append(0.0)
                continue
            # Average precision: mean of precision-at-each-relevant-rank
            precisions = [(i + 1) / (rank + 1) for i, rank in enumerate(hits)]
            scores.append(float(np.mean(precisions)))
        return float(np.mean(scores))
    # ── [END NEW METRIC] ─────────────────────────────────────────────────────

    def ndcg(self, relevance: np.ndarray, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        Primary metric in BEIR papers. Already in v1.
        """
        k = min(k, relevance.shape[1])
        scores = []
        for row in relevance:
            dcg = self._dcg(row[:k])
            ideal = np.sort(row)[::-1][:k]
            idcg = self._dcg(ideal)
            scores.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(scores))

    def _dcg(self, relevance: np.ndarray) -> float:
        gains = np.power(2, relevance) - 1
        discounts = np.log2(np.arange(len(relevance)) + 2)
        return float(np.sum(gains / discounts))

    # ── [ENHANCED] ───────────────────────────────────────────────────────────
    def compute_all(
        self,
        relevance: np.ndarray,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict:
        """
        Compute all metrics. Now includes MAP@10.

        Returns a flat dict suitable for LaTeX table generation:
          {
            "recall@1": 0.XXX, "recall@5": 0.XXX, "recall@10": 0.XXX,
            "mrr": 0.XXX, "ndcg@10": 0.XXX,
            "map@10": 0.XXX,    ← [NEW]
            "precision@5": 0.XXX, ...
          }
        """
        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(relevance, k)
            metrics[f"precision@{k}"] = self.precision_at_k(relevance, k)
        metrics["mrr"]    = self.mrr(relevance)
        metrics["ndcg@10"] = self.ndcg(relevance, k=10)
        metrics["map@10"] = self.map_at_k(relevance, k=10)   # [NEW]
        return metrics

    # ── [NEW METHOD] ─────────────────────────────────────────────────────────
    def per_query_scores(
        self,
        relevance: np.ndarray,
        k: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Return per-query scores for R@k, MRR, and MAP@k.

        Enables:
          - Per-difficulty-level breakdown (Easy/Medium/Hard)
          - Bootstrap CI computation over paired differences
          - Failure analysis (identify bottom-10% queries)

        Returns:
          {
            "recall@k": [0.0, 1.0, 1.0, ...],   # per-query binary R@k
            "mrr":      [0.0, 0.5, 1.0, ...],   # per-query RR
            "map@k":    [0.0, 0.75, 1.0, ...],  # per-query AP@k
          }
        """
        k = min(k, relevance.shape[1])
        r_at_k, mrr_scores, map_scores = [], [], []

        for row in relevance:
            # R@k: did any relevant item appear in top-k?
            r_at_k.append(float(np.max(row[:k])))

            # MRR: 1/rank of first hit
            hits = np.where(row == 1)[0]
            mrr_scores.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)

            # MAP@k
            hits_topk = np.where(row[:k] == 1)[0]
            if len(hits_topk) == 0:
                map_scores.append(0.0)
            else:
                prec = [(i + 1) / (rank + 1) for i, rank in enumerate(hits_topk)]
                map_scores.append(float(np.mean(prec)))

        return {
            f"recall@{k}": r_at_k,
            "mrr": mrr_scores,
            f"map@{k}": map_scores,
        }
    # ── [END NEW METHOD] ─────────────────────────────────────────────────────

    # ── [NEW METHOD] ─────────────────────────────────────────────────────────
    def format_table_row(
        self,
        method_name: str,
        metrics: Dict,
        best_baseline: Optional[Dict] = None,
    ) -> str:
        """
        Format a metrics dict as a LaTeX-ready table row.

        If best_baseline is provided, appends absolute Δ columns.
        Example output:
          StrucChunk + Hybrid & 0.730 & 0.890 & 0.910 & 0.796 & 0.862 & 0.831 \\\\
        """
        cols = [
            metrics.get("recall@1", 0),
            metrics.get("recall@5", 0),
            metrics.get("recall@10", 0),
            metrics.get("mrr", 0),
            metrics.get("ndcg@10", 0),
            metrics.get("map@10", 0),  # [NEW]
        ]
        row = f"{method_name:<28} & " + " & ".join(f"{v:.3f}" for v in cols)

        if best_baseline:
            delta_r5 = metrics.get("recall@5", 0) - best_baseline.get("recall@5", 0)
            pct = delta_r5 / best_baseline.get("recall@5", 1) * 100
            row += f" & {delta_r5:+.3f} ({pct:+.1f}\\%)"

        return row + " \\\\"
    # ── [END NEW METHOD] ─────────────────────────────────────────────────────
