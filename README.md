# StrucChunk: Structure-Aware Retrieval for Legal Document RAG

**Surya Deep Singh, Ashutosh Tripathi, Gaurav Deep Singh**

This repository contains the implementation and evaluation code for "StrucChunk: Structure-Aware Retrieval for Legal Document RAG."

## Overview

We propose **StrucChunk** (Structure-Aware Chunking) combined with hybrid dense-sparse retrieval for hierarchically structured legal documents. Our method:

- Treats documents as directed acyclic graphs (DAGs) of provisions
- Preserves clause boundaries and resolves cross-references prior to embedding
- Combines BM25 keyword precision with BGE-M3 semantic representations via Reciprocal Rank Fusion

## Key Results

- **LegalBench-RAG** (776 queries, English): Char-Recall@8 = 48.18% vs. 23.59% baseline (+104%)
- **CrPC** (501 queries, Indian criminal law): Recall@5 = 0.804 vs. 0.557 baseline (+44.4%)
- **CPC** (501 queries, Indian civil law): Recall@5 = 0.259 vs. 0.158 baseline (+64.6%)
- **GDPR** (501 queries, EU regulation): Recall@5 = 0.868 vs. 0.818 baseline (+6.1%)
- **NitiBench-CCL** (3,730 queries, Thai financial law): HR@5 = 0.856 vs. 0.594 naive (+44.1%)

All improvements are statistically significant (p ≤ 0.002, Wilcoxon signed-rank with Benjamini-Hochberg correction; Cohen's d 0.14–0.69).

## Repository Structure

```
.
├── src/
│   ├── parsers/
│   │   └── pdf_parser.py          # Legal PDF structure extraction
│   ├── chunkers/
│   │   ├── strucchunk.py          # StrucChunk implementation
│   │   └── recursive_chunker.py   # Baseline recursive chunker
│   ├── retrievers/
│   │   ├── dense_retriever.py     # BGE-M3 dense retrieval
│   │   ├── sparse_retriever.py    # BM25 sparse retrieval
│   │   └── hybrid_retriever.py    # RRF hybrid fusion
│   ├── evaluation/
│   │   ├── metrics.py             # Retrieval metrics (Recall@k, MRR, nDCG, MAP)
│   │   ├── chunking_quality.py    # Intrinsic chunking metrics
│   │   └── statistical_tests.py   # Wilcoxon + Benjamini-Hochberg
│   └── benchmarks/
│       └── legalbench_rag_evaluator.py  # LBR char-span evaluation
├── data/
│   ├── gdpr_2016_679.pdf                # GDPR source document
│   ├── the_code_of_criminal_procedure_1973.pdf  # CrPC source
│   ├── the_code_of_civil_procedure_1908.pdf     # CPC source
│   ├── gdpr_evaluation_queries.json     # 501 GDPR queries
│   ├── crpc_evaluation_queries.json     # 501 CrPC queries
│   ├── cpc_evaluation_queries.json      # 501 CPC queries
│   └── legalbench_rag/                  # LBR corpus (download separately)
├── run_experiments.py              # Main experiment runner (CrPC/CPC/GDPR)
├── run_legalbench_rag.py          # LegalBench-RAG evaluation
├── run_nitibench_ccl.py           # NitiBench-CCL evaluation
├── run_chunking_quality.py        # Intrinsic chunking metrics
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Requirements

- Python 3.10+
- 32GB RAM recommended (for BGE-M3 embeddings on large corpora)
- No GPU required (CPU-only inference)

### Setup

```bash
# Clone repository
git clone https://github.com/SinghSuryaDeep/legal-rag-strucchunk-1.git
cd legal-rag-strucchunk-1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for sentence splitting)
python -c "import nltk; nltk.download('punkt')"
```

## Data Preparation

### Layer 1: LegalBench-RAG

Download from the official repository:

```bash
# Clone LegalBench-RAG data
git clone https://github.com/zeroentropy-ai/legalbenchrag.git data/legalbench_rag
```

### Layer 2 & 3: CrPC, CPC, GDPR

Source PDFs are included in `data/`:
- `the_code_of_criminal_procedure_1973.pdf` (CrPC)
- `the_code_of_civil_procedure_1908.pdf` (CPC)
- `gdpr_2016_679.pdf` (GDPR)

Evaluation queries are provided in JSON format:
- `data/crpc_evaluation_queries.json` (501 queries)
- `data/cpc_evaluation_queries.json` (501 queries)
- `data/gdpr_evaluation_queries.json` (501 queries)

### Layer 4: NitiBench-CCL

Download from HuggingFace:

```bash
# Clone NitiBench repository
git clone https://github.com/vistec-ai/nitibench.git nitibench-main

# Data is automatically loaded from HuggingFace dataset VISAI-AI/nitibench
```

## Running Experiments

### CrPC Experiments (Layer 2)

```bash
# Full evaluation with ablation and chunking quality analysis
nohup python3 run_experiments.py \
  --pdf_path data/the_code_of_criminal_procedure_1973.pdf \
  --queries_path data/crpc_evaluation_queries.json \
  --ablation --chunking_quality \
  > logs/crpc_experiments.log 2>&1 &
```

### CPC Experiments (Layer 2)

```bash
nohup python3 run_experiments.py \
  --pdf_path data/the_code_of_civil_procedure_1908.pdf \
  --queries_path data/cpc_evaluation_queries.json \
  --ablation --chunking_quality \
  > logs/cpc_experiments.log 2>&1 &
```

### GDPR Experiments (Layer 3)

```bash
nohup python3 run_experiments.py \
  --pdf_path data/gdpr_2016_679.pdf \
  --queries_path data/gdpr_evaluation_queries.json \
  --ablation --chunking_quality \
  > logs/gdpr_experiments.log 2>&1 &
```

### LegalBench-RAG (Layer 1)

```bash
# Full evaluation (776 queries, ~3-4 hours)
nohup python3 run_legalbench_rag.py \
  --data_dir data/legalbench_rag \
  --n_queries 776 \
  > logs/legalbench_rag.log 2>&1 &

# Quick test (50 queries, ~15 minutes)
python3 run_legalbench_rag.py \
  --data_dir data/legalbench_rag \
  --n_queries 50
```

### NitiBench-CCL (Layer 4)

```bash
# Three-way chunking comparison (3,730 queries, ~2 hours)
nohup python3 run_nitibench_ccl.py \
  > logs/nitibench_ccl.log 2>&1 &
```

## Evaluation Metrics

### Retrieval Metrics (Layers 1-4)
- **Recall@k**: Fraction of queries with at least one relevant document in top-k
- **MRR**: Mean Reciprocal Rank of first relevant document
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **MAP@10**: Mean Average Precision at rank 10
- **HR@k**: Hit Rate at k (equivalent to Recall@k for single-label queries)

### Chunking Quality Metrics (Layers 2-3)
- **Boundary Precision**: Fraction of chunk boundaries aligned with section delimiters
- **Cross-Reference Preservation**: Fraction of cross-references co-located with source
- **Chunk Count Efficiency**: Section coverage per chunk
- **Semantic Coherence**: Intra-chunk similarity (future work)

### Statistical Tests
- Wilcoxon Signed-Rank Test (two-sided, paired per query)
- Benjamini-Hochberg FDR correction (α = 0.05)
- Cohen's d effect sizes
- 95% bootstrap confidence intervals

## Results Output

Results are saved to `results/` directory:

```
results/
├── crpc/
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── stats_YYYYMMDD_HHMMSS.json
│   └── chunking_quality.json
├── cpc/
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── stats_YYYYMMDD_HHMMSS.json
│   └── chunking_quality.json
├── gdpr/
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── stats_YYYYMMDD_HHMMSS.json
│   └── chunking_quality.json
├── legalbench_rag/
│   ├── lbr_metrics_YYYYMMDD_HHMMSS.json
│   └── lbr_stats_YYYYMMDD_HHMMSS.json
└── nitibench_ccl_results.json
```

## Implementation Details

### Chunking Parameters
- `max_tokens`: 512 (following standard RAG practice)
- `overlap`: 0.15 (15% overlap for recursive baseline)
- `max_section_multiplier`: 1.5 (sections up to 768 tokens kept whole)

### Retrieval Configuration
- **Dense**: BAAI/bge-m3 (567M params, 1024 dims, multilingual)
- **Sparse**: BM25 via bm25s v0.2.0
- **Fusion**: Reciprocal Rank Fusion (k=60)
- **Vector DB**: FAISS IndexFlatIP (L2-normalized)

### Cross-Reference Patterns
- Indian codes: `Section \d+`, `Order [IVX]+`, `Rule \d+`
- GDPR: `Article \d+(\(\d+\))?`, `Recital \d+`
- Thai: `มาตรา \d+(?:/\d+)?` (loaded from dataset annotations)

## Citation

```bibtex
@article{singh2026strucchunk,
  title={StrucChunk: Structure-Aware Retrieval for Legal Document RAG},
  author={Singh, Surya Deep and Tripathi, Ashutosh and Singh, Gaurav Deep},
  journal={Soft Computing},
  year={2026},
  note={Under review}
}
```

## License

Code released under MIT License. Evaluation datasets follow their respective licenses:
- LegalBench-RAG: Apache 2.0
- NitiBench-CCL: CC BY-SA 4.0
- Source legal documents: Public domain (official government publications)

## Acknowledgments

We thank the creators of:
- LegalBench-RAG (Pipitone & Alami, 2024)
- NitiBench (Akarajaradwong et al., 2025)
- BAAI/bge-m3 embedding model
- LegalBench benchmark (Guha et al., 2023)

## Contact

For questions, please contact the corresponding author: suryadeepsingh95@gmail.com