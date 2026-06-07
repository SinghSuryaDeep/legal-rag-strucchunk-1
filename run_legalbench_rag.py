"""
LegalBench-RAG Evaluation Runner
=================================
Evaluates StrucChunk on the LegalBench-RAG benchmark using character-span matching.

Usage:
  python run_legalbench_rag.py --data_dir data/legalbench_rag --n_queries 776

Download data from: https://github.com/zeroentropy-ai/legalbenchrag
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LegalBench-RAG evaluation for StrucChunk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (776 queries, ~3-4 hours)
  python run_legalbench_rag.py --data_dir data/legalbench_rag --n_queries 776

  # Quick smoke-test (50 queries, ~15 minutes)
  python run_legalbench_rag.py --data_dir data/legalbench_rag --n_queries 50

Data directory must contain:
  tests.json (or queries.json)  — LBR query file
  **/*.txt                      — source contract documents
""",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to LegalBench-RAG data directory",
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=776,
        help="Number of queries to evaluate (default: 776 = full LBR-mini)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/legalbench_rag",
        help="Directory for output JSON files (default: results/legalbench_rag)",
    )
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-m3",
        help="Sentence embedding model (default: BAAI/bge-m3)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--cache_dir",
        default="cache/lbr_embeddings",
        help="Embedding cache directory (default: cache/lbr_embeddings)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    args = parser.parse_args()

    # Validate data dir early
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error(
            "Download LegalBench-RAG from: https://github.com/zeroentropy-cc/legalbenchrag"
        )
        sys.exit(1)

    # Import evaluator (lazy, so missing deps give a clean error message)
    try:
        from legalbench_rag_evaluator import LegalBenchRAGEvaluator
    except ImportError:
        # Fallback to src/benchmarks location
        from src.benchmarks.legalbench_rag_evaluator import LegalBenchRAGEvaluator

    evaluator = LegalBenchRAGEvaluator(
        data_dir=args.data_dir,
        n_queries=args.n_queries,
        random_seed=args.seed,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        cache_dir=args.cache_dir,
    )

    logger.info(f"Starting LegalBench-RAG evaluation: {args.n_queries} queries")
    results = evaluator.run_evaluation(output_dir=args.output_dir)

    print(f"\n✅ LegalBench-RAG evaluation complete. Results saved to: {args.output_dir}")
    return results


if __name__ == "__main__":
    main()
