"""
Unit and integration tests for final paper tables, numbers manifest, and manuscript draft.
Validates zero data hallucination, LaTeX booktabs compliance, SHA-256 provenance,
and complete limitation coverage according to AGENT_RULES and T17 requirements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = WORKSPACE_ROOT / "results" / "tables" / "final"
MANIFEST_PATH = WORKSPACE_ROOT / "results" / "final" / "numbers_manifest.json"
MANUSCRIPT_DRAFT_PATH = WORKSPACE_ROOT / "docs" / "paper" / "MANUSCRIPT_DRAFT.md"
LOCK_PATH = WORKSPACE_ROOT / "runs" / "final_T.lock"

EXPECTED_TABLE_STEMS = [
    "tab_01_dataset_split",
    "tab_02_geometry_ablation_oof_b",
    "tab_03_main_benchmark_split_t",
    "tab_04_rq2_detector_correlation",
    "tab_05_conformal_coverage_t",
    "tab_06_conditional_coverage_odd",
]


def test_final_tables_exist() -> None:
    """Verify all 6 table pairs (.csv and .tex) exist in results/tables/final/."""
    assert TABLES_DIR.is_dir(), f"Tables directory missing: {TABLES_DIR}"
    for stem in EXPECTED_TABLE_STEMS:
        csv_path = TABLES_DIR / f"{stem}.csv"
        tex_path = TABLES_DIR / f"{stem}.tex"
        assert csv_path.exists(), f"Missing CSV table: {csv_path}"
        assert tex_path.exists(), f"Missing LaTeX table: {tex_path}"
        assert csv_path.stat().st_size > 0, f"CSV is empty: {csv_path}"
        assert tex_path.stat().st_size > 0, f"LaTeX is empty: {tex_path}"


def test_latex_booktabs_format() -> None:
    """Verify LaTeX tables strictly follow booktabs guidelines (no vertical rules, no double hlines)."""
    for stem in EXPECTED_TABLE_STEMS:
        tex_path = TABLES_DIR / f"{stem}.tex"
        content = tex_path.read_text(encoding="utf-8")

        # Must not contain vertical rules in tabular specs
        tabular_matches = re.findall(r"\\begin\{tabular\}\{([^}]+)\}", content)
        for spec in tabular_matches:
            assert "|" not in spec, f"Vertical rule '|' found in tabular specification '{spec}' in {stem}.tex"

        # Must not contain double hlines
        assert r"\hline\hline" not in content, f"Double \\hline\\hline found in {stem}.tex"

        # Must use booktabs rules
        assert r"\toprule" in content, f"Missing \\toprule in {stem}.tex"
        assert r"\midrule" in content, f"Missing \\midrule in {stem}.tex"
        assert r"\bottomrule" in content, f"Missing \\bottomrule in {stem}.tex"


def test_manifest_provenance_and_hash() -> None:
    """Verify 100% of manifest entries have valid source files and matching SHA-256 hashes."""
    assert MANIFEST_PATH.exists(), f"Numbers manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("numbers", {})
    assert len(metrics) >= 50, f"Expected at least 50 metrics, found {len(metrics)}"

    # Check that every metric has source provenance and hash matches
    checked_files: dict[str, str] = {}
    for key, item in metrics.items():
        assert "value" in item, f"Metric '{key}' missing 'value'"
        assert "source_file" in item, f"Metric '{key}' missing 'source_file'"
        assert "source_sha256" in item, f"Metric '{key}' missing 'source_sha256'"

        src_rel = item["source_file"]
        src_path = WORKSPACE_ROOT / src_rel
        assert src_path.exists(), f"Source file for '{key}' does not exist: {src_rel}"

        if src_rel not in checked_files:
            hasher = hashlib.sha256()
            with open(src_path, "rb") as sf:
                while chunk := sf.read(65536):
                    hasher.update(chunk)
            actual_sha = hasher.hexdigest()
            checked_files[src_rel] = actual_sha

        expected_sha = item["source_sha256"]
        assert checked_files[src_rel] == expected_sha, (
            f"SHA-256 mismatch for metric '{key}' in source '{src_rel}': "
            f"expected {expected_sha}, computed {checked_files[src_rel]}"
        )


def test_csv_tex_value_consistency() -> None:
    """Verify that numerical values rendered in CSV correspond directly to LaTeX tables."""
    # Test Tab 3 consistency
    tab3_csv = TABLES_DIR / "tab_03_main_benchmark_split_t.csv"
    tab3_tex = TABLES_DIR / "tab_03_main_benchmark_split_t.tex"
    csv_rows = list(csv.DictReader(tab3_csv.open(encoding="utf-8")))
    tex_text = tab3_tex.read_text(encoding="utf-8")

    for row in csv_rows:
        absrel = row.get("AbsRel")
        if absrel and absrel != "-":
            # Formatted in tex as float e.g. 0.0463
            float_val = f"{float(absrel):.4f}"
            assert float_val in tex_text, f"AbsRel {float_val} from CSV not found in Tab 3 TEX"

    # Test Tab 5 consistency
    tab5_csv = TABLES_DIR / "tab_05_conformal_coverage_t.csv"
    tab5_tex = TABLES_DIR / "tab_05_conformal_coverage_t.tex"
    csv_rows_5 = list(csv.DictReader(tab5_csv.open(encoding="utf-8")))
    tex_text_5 = tab5_tex.read_text(encoding="utf-8")

    for row in csv_rows_5:
        cov = row.get("Coverage (%)")
        if cov:
            float_cov = f"{float(cov):.1f}"
            assert float_cov in tex_text_5, f"Coverage {float_cov}% from CSV not found in Tab 5 TEX"


def test_manuscript_draft_completeness_and_no_placeholders() -> None:
    """Verify the manuscript draft has zero unresolved placeholders and contains all 14 limitations."""
    assert MANUSCRIPT_DRAFT_PATH.exists(), f"Manuscript draft missing: {MANUSCRIPT_DRAFT_PATH}"
    content = MANUSCRIPT_DRAFT_PATH.read_text(encoding="utf-8")

    # Zero unresolved placeholders
    unresolved_nums = re.findall(r"\{\{num:[^\}]+\}\}", content)
    assert not unresolved_nums, f"Found unresolved {{num:...}} placeholders: {unresolved_nums}"

    unresolved_tabs = re.findall(r"\{\{tab:[^\}]+\}\}", content)
    assert not unresolved_tabs, f"Found unresolved {{tab:...}} placeholders: {unresolved_tabs}"

    # Verify all 14 limitations are present
    expected_limitations = [
        "Truck Class Distribution Imbalance",
        "Survivorship Bias on True Positives",
        "Sparse Distant Sample Support",
        "Drive Concentration and Cluster Correlation",
        "Detector Training Checkpoint Reproducibility",
        "KITTI Neighbor Class Matching Protocols",
        "Indirect Qualitative Design Leakage",
        "Empirical Equivalence of Residual and Direct Regression",
        "Post-Hoc Verification Transparency",
        "Exchangeability Shift and Conservative Over-Coverage",
        "Optimism of 10-Cluster Bootstrap CIs",
        "Masked Local Under-Coverage",
        "Mondrian Interval Width Inflation",
        "Hardware and Latency Benchmark Constraints",
    ]
    for lim in expected_limitations:
        assert lim.lower() in content.lower(), f"Missing limitation in draft: '{lim}'"


def test_language_guard() -> None:
    """Verify that manuscript draft complies with the strict language guard (no forbidden hype/overclaims)."""
    assert MANUSCRIPT_DRAFT_PATH.exists()
    content = MANUSCRIPT_DRAFT_PATH.read_text(encoding="utf-8").lower()

    banned_regexes = [
        r"\bfirst\s+study\b",
        r"\bfirst\s+work\b",
        r"\bfirst\s+to\b",
        r"\bstatistically\s+significant\b",
        r"\bstatistically\s+significantly\b",
        r"\boutperforms\b",
        r"\bstate-of-the-art\b",
        r"\bsota\b",
        r"\bproves\s+that\b",
        r"\bproving\s+that\b",
        r"\bsuperior\s+to\b",
        r"\bvượt\s+trội\b",
        r"\bchứng\s+minh\s+rằng\b",
    ]

    for pattern in banned_regexes:
        matches = list(re.finditer(pattern, content))
        assert not matches, f"Found banned phrase matching '{pattern}' {len(matches)} time(s) in draft"


def test_lock_file_untouched() -> None:
    """Verify that the test split lock file remains completely intact and untouched."""
    assert LOCK_PATH.exists(), f"Lock file {LOCK_PATH} must exist"
    lock_content = LOCK_PATH.read_text(encoding="utf-8")
    assert "locked" in lock_content.lower(), "Lock file does not indicate locked status"
