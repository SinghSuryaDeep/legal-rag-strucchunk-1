"""
Statistical Significance Tests
==============================
Statistical tests for comparing retrieval methods.

CHANGES FROM v1:
  [BUG FIX]    p-values of "0.0000" replaced with "≤0.0001". A p-value of
               exactly 0 does not exist; Python's scipy.stats.wilcoxon returns
               values near machine epsilon (~5e-324) when the difference is
               overwhelming. Displaying "0.0000" signals careless reporting
               to any reviewer who knows statistics.
  [NEW]        format_p_value() — canonical display function used everywhere.
  [ENHANCED]   wilcoxon_test() output dict now includes 'p_display' field,
               so callers don't need to re-format.
  [NEW]        full_comparison_block() — generates the complete statistical
               block for one method-pair: p, significance, Cohen's d, 95% CI.
               Drop this directly into LaTeX Table 8.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ── [NEW] ────────────────────────────────────────────────────────────────────
def format_p_value(p: float) -> str:
    """
    Canonical p-value display for publication tables.

    CHANGE FROM v1: v1 printed raw floats, which displayed as "0.0000" for
    very small p-values. This is scientifically wrong (p=0 doesn't exist)
    and immediately flags the paper as having unreviewed statistics.

    Rules (standard in Nature/Science/NeurIPS):
      p < 0.0001  →  "≤0.0001"
      p < 0.001   →  "0.0009" (4 decimal places)
      p < 0.01    →  "0.009"  (3 decimal places)
      else        →  "0.XXX"  (3 decimal places)
    """
    if p < 0.0001:
        return "≤0.0001"
    elif p < 0.001:
        return f"{p:.4f}"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"
# ── [END NEW] ─────────────────────────────────────────────────────────────────


class StatisticalTests:
    """
    Statistical significance tests for retrieval experiments.

    Covers:
      Wilcoxon Signed-Rank Test   — non-parametric, paired, distribution-free
      Benjamini-Hochberg FDR      — multiple comparison correction
      Bootstrap CIs               — confidence intervals for effect magnitude
      Cohen's d                   — effect size (pairs with significance)
    """

    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alternative: str = "two-sided",
    ) -> Dict:
        """
        Paired Wilcoxon Signed-Rank Test.

        WHY WILCOXON over paired t-test: per-query Recall@5 scores are
        binary (0 or 1), strongly violating the normality assumption of the
        t-test. Wilcoxon makes no distributional assumption.

        CHANGES FROM v1:
          [ENHANCED] Returns 'p_display' field pre-formatted via format_p_value().
          [ENHANCED] Catches the case where all differences are zero and returns
                     p=1.0 instead of raising ValueError.
        """
        scores_a = np.array(scores_a, dtype=float)
        scores_b = np.array(scores_b, dtype=float)
        differences = scores_a - scores_b

        # [ENHANCED] Handle all-zero differences gracefully
        non_zero = differences[differences != 0]
        if len(non_zero) < 5:
            logger.warning(
                f"Only {len(non_zero)} non-zero differences found. "
                "Wilcoxon test may not be reliable."
            )
            if len(non_zero) == 0:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "p_display": "1.000",          # [ENHANCED]
                    "mean_diff": 0.0,
                    "significant": False,
                }

        try:
            statistic, p_value = stats.wilcoxon(
                scores_a, scores_b,
                alternative=alternative,
                zero_method="wilcox",
            )
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "p_display": "1.000",              # [ENHANCED]
                "mean_diff": float(np.mean(differences)),
                "significant": False,
            }

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "p_display": format_p_value(float(p_value)),  # [NEW FIELD]
            "mean_diff": float(np.mean(differences)),
            "significant": float(p_value) < 0.05,
        }

    def benjamini_hochberg(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> List[float]:
        """
        Benjamini-Hochberg FDR correction for multiple comparisons.

        We run 3 comparisons per dataset (vs Recursive+Dense, vs BM25, vs
        Naive Hybrid) × 3 datasets = 9 tests. Without correction, type-I
        error rate accumulates. BH controls the False Discovery Rate at α=0.05.

        No changes from v1 — implementation was correct.
        """
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        corrected = np.zeros(n)

        for i in range(n):
            corrected[sorted_idx[i]] = min(sorted_p[i] * n / (i + 1), 1.0)

        # Enforce monotonicity (Benjamini-Hochberg step-up)
        for i in range(n - 2, -1, -1):
            corrected[sorted_idx[i]] = min(
                corrected[sorted_idx[i]],
                corrected[sorted_idx[i + 1]] if i + 1 < n else 1.0,
            )

        return corrected.tolist()

    def bootstrap_ci(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for the mean of paired differences.

        Usage: pass (StrucChunk_scores - Baseline_scores) as `scores`.
        The CI tells reviewers the true improvement lies in [lower, upper]
        with 95% probability.

        No changes from v1.
        """
        np.random.seed(seed)
        scores = np.array(scores)
        bootstrap_means = [
            np.mean(np.random.choice(scores, size=len(scores), replace=True))
            for _ in range(n_bootstrap)
        ]
        alpha = 1 - confidence
        lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
        upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
        return lower, upper

    def effect_size_cohens_d(
        self,
        scores_a: List[float],
        scores_b: List[float],
    ) -> float:
        """
        Cohen's d for paired samples.

        Interpretation (Cohen 1988):
          d = 0.2  →  small   (noticeable only in large studies)
          d = 0.5  →  medium  (practically meaningful)
          d = 0.8  →  large   (clearly significant in practice)
          d > 1.0  →  very large (effect dominates)

        Given CrPC Recall@5 improvement of +0.24 (0.65→0.89),
        expect d >> 1.0, indicating a very large, practically dominant effect.

        No changes from v1.
        """
        diff = np.array(scores_a) - np.array(scores_b)
        std = np.std(diff, ddof=1)
        if std == 0:
            return float('inf') if np.mean(diff) > 0 else 0.0
        return float(np.mean(diff) / std)

    # ── [NEW METHOD] ─────────────────────────────────────────────────────────
    def full_comparison_block(
        self,
        our_scores: List[float],
        baseline_scores: List[float],
        comparison_label: str,
        dataset_label: str,
        seed: int = 42,
    ) -> Dict:
        """
        Generate the complete statistical block for one method-pair.

        This is what goes into Table 8 (Statistical Significance):
          | Comparison    | Dataset | p (BH-corr) | Cohen's d | 95% CI (Δ)     |
          |---------------|---------|-------------|-----------|-----------------|
          | vs Rec+Dense  | CrPC    | ≤0.0001     | 1.82      | [0.180, 0.300]  |

        Returns a dict with all fields pre-formatted for LaTeX.

        CHANGE FROM v1: Previously callers had to assemble this from three
        separate function calls. This method provides a single call returning
        everything needed for one table row.
        """
        wilcox = self.wilcoxon_test(our_scores, baseline_scores)
        d = self.effect_size_cohens_d(our_scores, baseline_scores)
        ci_lower, ci_upper = self.bootstrap_ci(
            [a - b for a, b in zip(our_scores, baseline_scores)],
            seed=seed,
        )
        mean_diff = np.mean(np.array(our_scores) - np.array(baseline_scores))
        pct_improvement = (
            mean_diff / max(np.mean(baseline_scores), 1e-9) * 100
        )

        # Cohen's d magnitude label
        if abs(d) >= 1.0:
            d_label = "very large"
        elif abs(d) >= 0.8:
            d_label = "large"
        elif abs(d) >= 0.5:
            d_label = "medium"
        else:
            d_label = "small"

        return {
            "comparison": comparison_label,
            "dataset": dataset_label,
            # Raw values for code use
            "p_value": wilcox["p_value"],
            "cohens_d": round(d, 3),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "mean_diff": round(float(mean_diff), 4),
            "pct_improvement": round(float(pct_improvement), 1),
            "significant": wilcox["significant"],
            # Pre-formatted strings for LaTeX table
            "p_display": wilcox["p_display"],               # [ENHANCED]
            "d_display": f"{d:.2f} ({d_label})",
            "ci_display": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            "sig_display": "Yes*" if wilcox["significant"] else "No",
            # LaTeX table row
            "latex_row": (
                f"{comparison_label} & {dataset_label} & "
                f"{wilcox['p_display']} & {d:.2f} & "
                f"[{ci_lower:.3f}, {ci_upper:.3f}] \\\\"
            ),
        }
    # ── [END NEW METHOD] ─────────────────────────────────────────────────────

    # ── [NEW METHOD] ─────────────────────────────────────────────────────────
    def run_full_significance_table(
        self,
        our_scores: List[float],
        baseline_scores_dict: Dict[str, List[float]],
        dataset_label: str,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Run all comparisons for one dataset and apply BH correction.

        Returns (comparison_blocks, corrected_p_values).
        Each block in comparison_blocks has 'corrected_p_display' added.

        Usage:
            blocks, _ = tests.run_full_significance_table(
                our_scores=strucchunk_r5,
                baseline_scores_dict={
                    "vs Rec+Dense":  recursive_r5,
                    "vs BM25":       bm25_r5,
                    "vs Naive Hybrid": naive_r5,
                },
                dataset_label="CrPC",
            )
        """
        blocks = []
        for label, baseline_scores in baseline_scores_dict.items():
            block = self.full_comparison_block(
                our_scores, baseline_scores, label, dataset_label, seed
            )
            blocks.append(block)

        # Apply Benjamini-Hochberg across all comparisons
        p_values = [b["p_value"] for b in blocks]
        corrected = self.benjamini_hochberg(p_values)

        for block, cp in zip(blocks, corrected):
            block["corrected_p_value"] = cp
            block["corrected_p_display"] = format_p_value(cp)     # [ENHANCED]
            block["corrected_significant"] = cp < 0.05
            block["corrected_sig_display"] = "Yes*" if cp < 0.05 else "No"
            # Update the latex row with corrected p
            block["latex_row"] = (
                f"{block['comparison']} & {block['dataset']} & "
                f"{block['corrected_p_display']} & {block['cohens_d']:.2f} & "
                f"[{block['ci_lower']:.3f}, {block['ci_upper']:.3f}] \\\\"
            )

        return blocks, corrected
    # ── [END NEW METHOD] ─────────────────────────────────────────────────────
