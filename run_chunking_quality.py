"""
Chunking Quality Analysis
=========================
Computes intrinsic chunking metrics independent of retrieval performance.

Metrics:
  - Boundary Precision: Fraction of chunk boundaries aligned with section delimiters
  - Cross-Reference Preservation: Fraction of cross-references co-located with source
  - Chunk Count Efficiency: Section coverage per chunk
  - Intra-Chunk Coherence: Average semantic similarity within chunks

Usage:
  # Quick analysis (no encoder)
  python run_chunking_quality.py --pdf_path data/crpc.pdf

  # Full analysis with coherence metric
  python run_chunking_quality.py --pdf_path data/crpc.pdf --use_encoder
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="[NEW] Compute intrinsic chunking quality metrics for StrucChunk paper Table 1"
    )
    parser.add_argument(
        "--pdf_path", required=True,
        help="Path to legal PDF (CrPC or CPC)"
    )
    parser.add_argument(
        "--output_dir", default="results/chunking_quality",
        help="Output directory for quality report"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512,
        help="Token limit per chunk"
    )
    parser.add_argument(
        "--use_encoder", action="store_true",
        help="Use BGE-M3 to compute intra-chunk coherence (slower but complete)"
    )
    parser.add_argument(
        "--both_documents", action="store_true",
        help="Process both CrPC and CPC automatically (requires both PDFs in data/)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.both_documents:
        pdfs = [
            Path("data/the_code_of_criminal_procedure_1973.pdf"),
            Path("data/the_code_of_civil_procedure_1908.pdf"),
        ]
    else:
        pdfs = [Path(args.pdf_path)]

    all_reports = {}
    for pdf_path in pdfs:
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}. Skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"CHUNKING QUALITY ANALYSIS — {pdf_path.name}")
        print(f"{'='*70}")

        from src.evaluation.chunking_quality import run_chunking_quality
        report = run_chunking_quality(
            pdf_path=str(pdf_path),
            document_name=pdf_path.stem,
            chunk_size=args.chunk_size,
            chunk_overlap=0.15,
            use_encoder=args.use_encoder,
        )

        all_reports[pdf_path.stem] = report

    # Save combined report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"chunking_quality_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n✅ Chunking quality analysis complete!")
    print(f"   Results saved to: {out_path}")
    print(f"\n   ➜ Use these numbers for Table 1 in paper_final.md")
    print(f"   ➜ Section heading: '5.1 Chunking Quality Analysis'")

    # Print the paper-ready claim sentence
    for doc_name, report in all_reports.items():
        bp = report.get("boundary_precision", {})
        cr = report.get("cross_ref_preservation", {})
        eff = report.get("chunk_count_efficiency", {})

        print(f"\n--- PAPER CLAIM FOR {doc_name.upper()} ---")
        print(
            f"\"StrucChunk achieves boundary precision of "
            f"{bp.get('strucchunk', 'X')}% "
            f"(vs {bp.get('recursive', 'X')}% for recursive splitting), "
            f"preserves {cr.get('strucchunk', {}).get('rate_pct', 'X')}% "
            f"of {cr.get('strucchunk', {}).get('total', 'X')} "
            f"cross-references, and achieves equivalent coverage with "
            f"{eff.get('chunk_reduction_pct', 'X')}% fewer chunks.\""
        )


if __name__ == "__main__":
    main()
