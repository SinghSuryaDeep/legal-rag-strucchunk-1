"""
StrucChunk Experiment Runner
=============================

Evaluates StrucChunk and baseline methods on legal document retrieval tasks.

Usage:
  # Standard evaluation
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json

  # With ablation study (2×2 chunking × retrieval matrix)
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json --ablation

  # With chunking quality metrics
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json --chunking_quality
"""

import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── [BUG FIX] Section number matching ────────────────────────────────────────
def section_found(text: str, sec: str) -> bool:
    """
    Check whether a section identifier appears in chunk text.

    BUG FIX FROM v1: v1 used `f"Section {sec}" in text`, which matches
    "Section 1673" when searching for "Section 167". Word-boundary regex
    prevents this false-positive.

    Handles both "Section 167" and "§167" notation.
    """
    patterns = [
        re.compile(rf'\bSection\s+{re.escape(sec)}\b', re.IGNORECASE),
        re.compile(rf'\b§\s*{re.escape(sec)}\b'),
        re.compile(rf'\bArticle\s+{re.escape(sec)}\b', re.IGNORECASE),  # GDPR
    ]
    return any(p.search(text) for p in patterns)
# ── [END BUG FIX] ─────────────────────────────────────────────────────────────


class ExperimentRunner:
    """
    Runs retrieval experiments comparing chunking strategies.

    Standard run: 4 methods (Recursive+Dense, BM25, Naive Hybrid, StrucChunk+Hybrid)
    Ablation run: 6 methods (adds StrucChunk+Dense, StrucChunk+BM25) [NEW]
    """

    def __init__(
        self,
        pdf_path: str,
        queries_path: str,
        output_dir: str = "results",
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 512,
        chunk_overlap: float = 0.15,
        random_seed: int = 42,
        run_ablation: bool = False,           # [NEW]
        run_chunking_quality: bool = False,   # [NEW]
    ):
        self.pdf_path = Path(pdf_path)
        self.queries_path = Path(queries_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.random_seed = random_seed
        self.run_ablation = run_ablation
        self.run_chunking_quality = run_chunking_quality
        np.random.seed(random_seed)

        with open(queries_path) as f:
            self.query_data = json.load(f)
        self.queries = self.query_data["queries"]
        logger.info(f"Loaded {len(self.queries)} evaluation queries")

    def run_all_experiments(self) -> Dict:
        from src.parsers.pdf_parser import LegalPDFParser
        from src.chunkers.recursive_chunker import RecursiveChunker
        from src.chunkers.strucchunk import StrucChunker
        from src.retrievers.dense_retriever import DenseRetriever
        from src.retrievers.sparse_retriever import SparseRetriever
        from src.retrievers.hybrid_retriever import HybridRetriever
        from src.evaluation.metrics import RetrievalMetrics
        from src.evaluation.statistical_tests import StatisticalTests

        # ── Step 1: Parse ──────────────────────────────────────────────────
        logger.info("\n[1/5] Parsing legal document...")
        parser = LegalPDFParser()
        document = parser.parse(self.pdf_path)
        logger.info(f"Parsed {len(document.sections)} sections")

        # ── Step 2: Chunk ──────────────────────────────────────────────────
        logger.info("\n[2/5] Creating chunks...")
        recursive_chunker = RecursiveChunker(
            chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        struc_chunker = StrucChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            add_breadcrumbs=True,
            resolve_cross_refs=True,
        )
        recursive_chunks = recursive_chunker.chunk(document.full_text)
        struc_chunks = struc_chunker.chunk(document)
        logger.info(f"  Recursive: {len(recursive_chunks)} chunks")
        logger.info(f"  StrucChunk: {len(struc_chunks)} chunks")

        # ── [NEW] Step 2b: Chunking quality analysis ───────────────────────
        if self.run_chunking_quality:
            logger.info("\n[2b/5] Computing chunking quality metrics...")
            from src.evaluation.chunking_quality import ChunkingQualityAnalyzer
            analyzer = ChunkingQualityAnalyzer(
                document, recursive_chunks, struc_chunks
            )
            quality_report = analyzer.run_full_analysis()
            analyzer.print_table(quality_report)

            # Save
            with open(self.output_dir / "chunking_quality.json", "w") as f:
                json.dump(quality_report, f, indent=2)

        # ── Step 3: Build retrievers ───────────────────────────────────────
        logger.info("\n[3/5] Building retrievers...")
        rec_texts = [c["text"] for c in recursive_chunks]
        struc_texts = [c["text"] for c in struc_chunks]

        # Standard 4 methods
        methods: Dict[str, Tuple] = {
            "Recursive + Dense": (
                DenseRetriever(chunks=rec_texts, model_name=self.embedding_model),
                recursive_chunks,
            ),
            "BM25 Only": (
                SparseRetriever(chunks=rec_texts),
                recursive_chunks,
            ),
            "Naive Hybrid": (
                HybridRetriever(chunks=rec_texts, model_name=self.embedding_model),
                recursive_chunks,
            ),
            "StrucChunk + Hybrid": (
                HybridRetriever(chunks=struc_texts, model_name=self.embedding_model),
                struc_chunks,
            ),
        }

        # [NEW] Ablation: add StrucChunk+Dense and StrucChunk+BM25
        if self.run_ablation:
            logger.info("  Building ablation methods (StrucChunk+Dense, StrucChunk+BM25)...")
            methods["StrucChunk + Dense"] = (
                DenseRetriever(chunks=struc_texts, model_name=self.embedding_model),
                struc_chunks,
            )
            methods["StrucChunk + BM25"] = (
                SparseRetriever(chunks=struc_texts),
                struc_chunks,
            )

        # ── Step 4: Evaluate ───────────────────────────────────────────────
        logger.info(f"\n[4/5] Evaluating {len(methods)} methods...")
        all_results = {}
        for method_name, (retriever, chunks) in methods.items():
            logger.info(f"  → {method_name}")
            all_results[method_name] = self._evaluate_retriever(retriever, chunks)

        # ── Step 5: Metrics + stats ────────────────────────────────────────
        logger.info("\n[5/5] Computing metrics and statistical tests...")
        metrics_calc = RetrievalMetrics()
        stat_calc = StatisticalTests()

        metrics = self._compute_all_metrics(all_results, metrics_calc)
        stat_tests = self._run_statistical_tests(all_results, metrics, stat_calc)

        self._save_and_print(metrics, stat_tests, all_results)

        return {"metrics": metrics, "statistical_tests": stat_tests}

    def _evaluate_retriever(
        self,
        retriever,
        chunks: List[Dict],
        k: int = 10,
    ) -> Dict:
        """
        Evaluate a retriever across all queries.

        BUG FIX: Uses section_found() with word-boundary regex instead of
        string `in` check. See module docstring.
        """
        results = {"factual": [], "clause_analysis": [], "cross_reference": []}

        for query in self.queries:
            query_text = query["query"]
            expected_sections = query["expected_sections"]
            query_type = query["type"]

            retrieved = retriever.retrieve(query_text, k=k)
            retrieved_texts = [r["text"] for r in retrieved]

            relevance_scores = []
            for sec in expected_sections:
                found_at = None
                for rank, text in enumerate(retrieved_texts):
                    # [BUG FIX] Word-boundary regex instead of string `in`
                    if section_found(text, sec):
                        found_at = rank + 1
                        break
                relevance_scores.append({
                    "section": sec,
                    "found": found_at is not None,
                    "rank": found_at,
                })

            results[query_type].append({
                "query_id": query["id"],
                "query": query_text,
                "expected_sections": expected_sections,
                "relevance": relevance_scores,
                "difficulty": query.get("difficulty", "medium"),  # [ENHANCED]
            })

        return results

    def _compute_all_metrics(
        self,
        all_results: Dict,
        metrics_calc,
    ) -> Dict:
        """
        Compute metrics for all methods.
        [ENHANCED] Now includes MAP@10 and per-difficulty breakdown.
        """
        computed = {}

        for method_name, method_results in all_results.items():
            all_rel = []
            by_type = {}
            by_difficulty = {}  # [NEW] for query difficulty analysis

            for query_type, type_results in method_results.items():
                type_rel = []
                for result in type_results:
                    rel_vector = self._build_rel_vector(result["relevance"])
                    type_rel.append(rel_vector)
                    all_rel.append(rel_vector)

                    # [NEW] Group by difficulty
                    diff = result.get("difficulty", "medium")
                    by_difficulty.setdefault(diff, []).append(rel_vector)

                arr = np.array(type_rel) if type_rel else np.zeros((1, 10))
                by_type[query_type] = {
                    "recall@1":  float(np.mean(arr[:, 0])),
                    "recall@5":  float(np.mean(np.max(arr[:, :5], axis=1))),
                    "recall@10": float(np.mean(np.max(arr[:, :10], axis=1))),
                    "mrr":       metrics_calc.mrr(arr),
                    "map@10":    metrics_calc.map_at_k(arr, k=10),  # [NEW]
                    "count":     len(type_results),
                }

            all_arr = np.array(all_rel) if all_rel else np.zeros((1, 10))
            computed[method_name] = {
                "overall": {
                    "recall@1":  float(np.mean(all_arr[:, 0])),
                    "recall@5":  float(np.mean(np.max(all_arr[:, :5], axis=1))),
                    "recall@10": float(np.mean(np.max(all_arr[:, :10], axis=1))),
                    "mrr":       metrics_calc.mrr(all_arr),
                    "ndcg@10":   metrics_calc.ndcg(all_arr, k=10),
                    "map@10":    metrics_calc.map_at_k(all_arr, k=10),  # [NEW]
                    "count":     len(all_rel),
                },
                "by_type": by_type,
                # [NEW] difficulty breakdown
                "by_difficulty": {
                    diff: {
                        "recall@5": float(np.mean(np.max(np.array(vecs)[:, :5], axis=1))),
                        "count": len(vecs),
                    }
                    for diff, vecs in by_difficulty.items()
                },
            }

        return computed

    def _build_rel_vector(self, relevance_scores: List[Dict]) -> List[int]:
        """Build binary relevance vector from per-section scores."""
        rel_vector = [0] * 10
        for rel in relevance_scores:
            if rel["found"] and rel["rank"] is not None:
                for i in range(rel["rank"] - 1, 10):
                    rel_vector[i] = 1
        return rel_vector

    def _run_statistical_tests(
        self,
        all_results: Dict,
        metrics: Dict,
        stat_calc,
    ) -> Dict:
        """
        [ENHANCED] Uses full_comparison_block() for pre-formatted LaTeX output.
        """
        our_method = "StrucChunk + Hybrid"
        if our_method not in all_results:
            return {}

        our_scores = self._get_r5_scores(all_results[our_method])
        baseline_scores = {}
        for baseline in ["Recursive + Dense", "BM25 Only", "Naive Hybrid"]:
            if baseline in all_results:
                baseline_scores[f"vs {baseline}"] = self._get_r5_scores(
                    all_results[baseline]
                )

        dataset_name = self.pdf_path.stem.replace("the_code_of_", "").upper()
        blocks, _ = stat_calc.run_full_significance_table(
            our_scores, baseline_scores, dataset_name
        )

        return {b["comparison"]: b for b in blocks}

    def _get_r5_scores(self, results: Dict) -> List[float]:
        scores = []
        for type_results in results.values():
            for result in type_results:
                found = any(
                    rel["found"] and rel["rank"] is not None and rel["rank"] <= 5
                    for rel in result["relevance"]
                )
                scores.append(1.0 if found else 0.0)
        return scores

    def _save_and_print(
        self,
        metrics: Dict,
        stat_tests: Dict,
        all_results: Dict,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.output_dir / f"metrics_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(self.output_dir / f"stats_{timestamp}.json", "w") as f:
            json.dump(stat_tests, f, indent=2)

        self._print_results_table(metrics)
        self._print_ablation_table(metrics)
        self._print_stats_table(stat_tests)
        self._print_latex_table(metrics)   # [NEW]

    def _print_results_table(self, metrics: Dict):
        print("\n" + "=" * 95)
        print("RESULTS SUMMARY — Overall Performance")
        print("=" * 95)
        # [ENHANCED] MAP@10 column added
        print(
            f"{'Method':<28} {'R@1':>7} {'R@5':>7} {'R@10':>7} "
            f"{'MRR':>7} {'nDCG@10':>9} {'MAP@10':>8}"
        )
        print("-" * 95)
        for method, m in metrics.items():
            o = m["overall"]
            print(
                f"{method:<28} {o['recall@1']:>7.3f} {o['recall@5']:>7.3f} "
                f"{o['recall@10']:>7.3f} {o['mrr']:>7.3f} "
                f"{o['ndcg@10']:>9.3f} {o['map@10']:>8.3f}"
            )
        print("-" * 95)

        # Print improvement vs best baseline
        if "StrucChunk + Hybrid" in metrics:
            our = metrics["StrucChunk + Hybrid"]["overall"]
            best_r5 = max(
                m["overall"]["recall@5"]
                for k, m in metrics.items()
                if k != "StrucChunk + Hybrid" and "StrucChunk" not in k
            )
            delta = our["recall@5"] - best_r5
            print(
                f"\n→ StrucChunk+Hybrid Recall@5: {our['recall@5']:.3f} "
                f"(best baseline: {best_r5:.3f}, Δ={delta:+.3f}, {delta/best_r5*100:+.1f}%)"
            )

        print("\nRecall@5 by query type:")
        print(f"{'Method':<28} {'Factual':>10} {'Clause':>10} {'Cross-Ref':>10} {'MAP@10':>8}")
        print("-" * 70)
        for method, m in metrics.items():
            bt = m.get("by_type", {})
            f = bt.get("factual", {}).get("recall@5", 0)
            c = bt.get("clause_analysis", {}).get("recall@5", 0)
            cr = bt.get("cross_reference", {}).get("recall@5", 0)
            ma = bt.get("factual", {}).get("map@10", 0)  # representative
            print(f"{method:<28} {f:>10.3f} {c:>10.3f} {cr:>10.3f} {ma:>8.3f}")

    def _print_ablation_table(self, metrics: Dict):
        """[NEW] Print 2×2 ablation matrix if ablation methods are present."""
        ablation_methods = [k for k in metrics if "StrucChunk + Dense" in k or "StrucChunk + BM25" in k]
        if not ablation_methods:
            return

        print("\n" + "=" * 70)
        print("2×2 ABLATION TABLE (Recall@5)")
        print("=" * 70)
        print(f"{'':30} {'Dense':>12} {'BM25':>12} {'Hybrid (RRF)':>14}")
        print("-" * 70)

        def r5(name):
            return metrics.get(name, {}).get("overall", {}).get("recall@5", 0.0)

        print(f"{'Recursive (baseline)':<30} {r5('Recursive + Dense'):>12.3f} "
              f"{r5('BM25 Only'):>12.3f} {r5('Naive Hybrid'):>14.3f}")
        print(f"{'StrucChunk (ours)':<30} {r5('StrucChunk + Dense'):>12.3f} "
              f"{r5('StrucChunk + BM25'):>12.3f} {r5('StrucChunk + Hybrid'):>14.3f}")

        # Chunking effect (keeping retrieval fixed at Hybrid)
        rec_hybrid = r5("Naive Hybrid")
        struc_hybrid = r5("StrucChunk + Hybrid")
        chunking_delta = struc_hybrid - rec_hybrid

        # Retrieval effect (keeping chunking fixed at StrucChunk)
        struc_dense = r5("StrucChunk + Dense")
        retrieval_delta = struc_hybrid - struc_dense

        print(f"\n  Chunking effect (R@5, StrucChunk vs Recursive, Hybrid fixed): "
              f"{chunking_delta:+.3f}")
        print(f"  Retrieval effect (R@5, Hybrid vs Dense, StrucChunk fixed):     "
              f"{retrieval_delta:+.3f}")
        print(f"  Total improvement (StrucChunk+Hybrid vs Recursive+Dense):      "
              f"{struc_hybrid - r5('Recursive + Dense'):+.3f}")

    def _print_stats_table(self, stat_tests: Dict):
        """[ENHANCED] Stats table with pre-formatted BH-corrected p-values."""
        if not stat_tests:
            return
        print("\n" + "=" * 90)
        print("STATISTICAL SIGNIFICANCE (Wilcoxon Signed-Rank + Benjamini-Hochberg FDR)")
        print("=" * 90)
        cohens_d_hdr = "Cohen's d"
        print(
            f"{'Comparison':<30} {'p (BH-corr)':>12} {cohens_d_hdr:>11} "
            f"{'95% CI':>16} {'Sig':>5} {'Δ%':>8}"
        )
        print("-" * 86)
        for comp, result in stat_tests.items():
            print(
                f"{comp:<30} "
                f"{result.get('corrected_p_display', result.get('p_display', 'N/A')):>12} "
                f"{result.get('cohens_d', 0.0):>11.3f} "
                f"{result.get('ci_display', 'N/A'):>16} "
                f"{result.get('corrected_sig_display', 'N/A'):>5} "
                f"{result.get('pct_improvement', 0.0):>8.1f}%"
            )
        print("* BH correction controls False Discovery Rate at α=0.05")

    # ── [NEW] LaTeX-ready output ──────────────────────────────────────────────
    def _print_latex_table(self, metrics: Dict):
        """
        [NEW] Print results in LaTeX tabular format for direct paste into paper.

        This saves the ~30 minutes of manually formatting numbers into LaTeX.
        Copy-paste into Table 4 (CrPC) or Table 5 (CPC).
        """
        print("\n" + "=" * 90)
        print("LATEX TABLE (copy into paper)")
        print("=" * 90)
        print("% R@1  R@5  R@10  MRR  nDCG@10  MAP@10")

        method_order = [
            "Recursive + Dense", "BM25 Only",
            "Naive Hybrid", "StrucChunk + Hybrid",
        ]

        best_baseline = None
        for m in ["Recursive + Dense", "BM25 Only", "Naive Hybrid"]:
            if m in metrics:
                if best_baseline is None or metrics[m]["overall"]["recall@5"] > best_baseline["recall@5"]:
                    best_baseline = metrics[m]["overall"]

        for method in method_order:
            if method not in metrics:
                continue
            o = metrics[method]["overall"]
            delta_r5 = o["recall@5"] - best_baseline.get("recall@5", 0) if best_baseline else 0
            pct = delta_r5 / best_baseline.get("recall@5", 1) * 100 if best_baseline else 0

            if "StrucChunk" in method:
                row = (
                    f"\\textbf{{{method}}} & "
                    f"\\textbf{{{o['recall@1']:.3f}}} & "
                    f"\\textbf{{{o['recall@5']:.3f}}} & "
                    f"\\textbf{{{o['recall@10']:.3f}}} & "
                    f"\\textbf{{{o['mrr']:.3f}}} & "
                    f"\\textbf{{{o['ndcg@10']:.3f}}} & "
                    f"\\textbf{{{o['map@10']:.3f}}} & "
                    f"+{delta_r5:.3f} ({pct:+.1f}\\%) \\\\"
                )
            else:
                row = (
                    f"{method} & {o['recall@1']:.3f} & {o['recall@5']:.3f} & "
                    f"{o['recall@10']:.3f} & {o['mrr']:.3f} & "
                    f"{o['ndcg@10']:.3f} & {o['map@10']:.3f} & — \\\\"
                )
            print(row)

        print(
            "\\midrule\n"
            "% ↑ Copy this block into Table 4 (CrPC) or Table 5 (CPC)\n"
            "% Column header: Method & R@1 & R@5 & R@10 & MRR & nDCG@10 & MAP@10 & Δ R@5\n"
        )
    # ── [END NEW] ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run StrucChunk retrieval experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 4-method evaluation
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json

  # With 2x2 ablation (adds StrucChunk+Dense and StrucChunk+BM25)
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json --ablation

  # With chunking quality analysis (Table 1 in paper)
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json --chunking_quality

  # Full run for paper submission
  python run_experiments.py --pdf_path data/crpc.pdf --queries_path data/crpc_queries.json --ablation --chunking_quality
"""
    )
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--queries_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--embedding_model", default="BAAI/bge-m3")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    # [NEW] Ablation flag
    parser.add_argument(
        "--ablation", action="store_true",
        help="[NEW] Run 2x2 ablation (adds StrucChunk+Dense and StrucChunk+BM25 methods)"
    )
    # [NEW] Chunking quality flag
    parser.add_argument(
        "--chunking_quality", action="store_true",
        help="[NEW] Compute intrinsic chunking quality metrics (boundary precision, etc.)"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        pdf_path=args.pdf_path,
        queries_path=args.queries_path,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        random_seed=args.seed,
        run_ablation=args.ablation,
        run_chunking_quality=args.chunking_quality,
    )

    results = runner.run_all_experiments()
    print("\n✅ Experiments complete. Results saved to:", args.output_dir)
    return results


if __name__ == "__main__":
    main()
