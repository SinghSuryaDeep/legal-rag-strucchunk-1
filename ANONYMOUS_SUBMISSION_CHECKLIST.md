# Anonymous Submission Checklist for 4open.science

## ✅ Files to Include

### Core Implementation
- [x] `src/` - All source code
  - `src/parsers/pdf_parser.py`
  - `src/chunkers/strucchunk.py`
  - `src/chunkers/recursive_chunker.py`
  - `src/retrievers/dense_retriever.py`
  - `src/retrievers/sparse_retriever.py`
  - `src/retrievers/hybrid_retriever.py`
  - `src/evaluation/metrics.py`
  - `src/evaluation/chunking_quality.py`
  - `src/evaluation/statistical_tests.py`
  - `src/benchmarks/legalbench_rag_evaluator.py`

### Experiment Runners
- [x] `run_experiments.py` - CrPC/CPC/GDPR evaluation
- [x] `run_legalbench_rag.py` - LegalBench-RAG evaluation
- [x] `run_nitibench_ccl.py` - NitiBench-CCL evaluation
- [x] `run_chunking_quality.py` - Intrinsic chunking metrics

### Data Files
- [x] `data/gdpr_2016_679.pdf` - GDPR source document
- [x] `data/the_code_of_criminal_procedure_1973.pdf` - CrPC source
- [x] `data/the_code_of_civil_procedure_1908.pdf` - CPC source
- [x] `data/gdpr_evaluation_queries.json` - 501 GDPR queries
- [x] `data/crpc_evaluation_queries.json` - 501 CrPC queries
- [x] `data/cpc_evaluation_queries.json` - 501 CPC queries

### Configuration
- [x] `requirements.txt` - Python dependencies
- [x] `README.md` - Anonymous documentation
- [x] `.gitignore` - Updated for anonymous submission

## ❌ Files to Exclude (via .gitignore)

### Personal/Draft Files
- [x] `MASTER_CHANGES*.md` - Personal change logs
- [x] `CHANGES.md` - Personal notes
- [x] `paper_*.md` - Paper drafts
- [x] `point_*.py` - Analysis scripts with personal notes
- [x] `*_draft*.md` - Draft documents
- [x] `*_notes*.md` - Personal notes

### Generated/Cache Files
- [x] `cache/` - Embedding caches
- [x] `logs/` - Experiment logs (keep structure, ignore content)
- [x] `results/` - Result files (keep structure, ignore content)
- [x] `__pycache__/` - Python cache
- [x] `*.pkl` - Pickle files
- [x] `.DS_Store` - macOS metadata

### Environment
- [x] `.env` - Environment variables
- [x] `venv/` - Virtual environment
- [x] `.vscode/` - Editor settings

## 🔍 Anonymization Verification

### Code Files
- [x] No author names in comments
- [x] No institutional affiliations in docstrings
- [x] No personal email addresses
- [x] No internal project names

### Documentation
- [x] README.md uses "Anonymous submission"
- [x] No author information in citation
- [x] Contact section refers to TMLR review system
- [x] Acknowledgments section is generic

### Data Files
- [x] Query JSON files contain no personal identifiers
- [x] Source PDFs are public domain documents
- [x] No proprietary or confidential data

## 📦 External Data Dependencies

### Must Download Separately
1. **LegalBench-RAG** (Layer 1)
   - Source: https://github.com/zeroentropy-ai/legalbenchrag
   - License: Apache 2.0
   - Size: ~500MB
   - Instructions in README.md

2. **NitiBench-CCL** (Layer 4)
   - Source: https://github.com/vistec-ai/nitibench
   - HuggingFace: VISAI-AI/nitibench
   - License: CC BY-SA 4.0
   - Size: ~100MB
   - Instructions in README.md

## 🚀 Reproducibility Requirements

### Included in Repository
- [x] Complete source code
- [x] All evaluation queries (1,503 queries for Layers 2-3)
- [x] Source legal documents (CrPC, CPC, GDPR PDFs)
- [x] Exact dependency versions in requirements.txt
- [x] Random seeds documented (seed=42)
- [x] Hyperparameters documented in README

### Documented but Not Included
- [x] LegalBench-RAG corpus (download instructions provided)
- [x] NitiBench-CCL corpus (download instructions provided)
- [x] Pre-trained models (BAAI/bge-m3 from HuggingFace)

## ✅ Final Checks Before Upload

1. **Run git status** to verify no sensitive files are tracked
2. **Test clone** in a fresh directory to verify completeness
3. **Run quick test** to verify dependencies install correctly:
   ```bash
   python3 run_experiments.py --pdf_path data/gdpr_2016_679.pdf \
     --queries_path data/gdpr_evaluation_queries.json
   ```
4. **Verify README** renders correctly on 4open.science
5. **Check file sizes** - ensure no large cache files are included

## 📋 Upload Instructions for 4open.science

1. Create account at https://anonymous.4open.science/
2. Create new repository
3. Upload via:
   - Git push (recommended)
   - Web interface
   - ZIP upload
4. Verify anonymous URL works
5. Include anonymous URL in TMLR submission

## 🔗 Anonymous URL Format

The anonymous URL will be:
```
https://anonymous.4open.science/r/[random-id]/
```

Include this URL in the "Code Availability" section of your TMLR submission.

---

**Last Updated**: 2026-06-07
**Status**: Ready for anonymous submission