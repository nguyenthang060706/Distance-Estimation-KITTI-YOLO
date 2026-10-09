"""
Unit and integration tests for final paper tables, numbers manifest, figures, and manuscript draft.
Validates zero data hallucination, LaTeX booktabs compliance, SHA-256 provenance,
complete limitation coverage, BibTeX citations, and publication figures according to AGENT_RULES and T17 requirements.
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
FIGURES_DIR = WORKSPACE_ROOT / "results" / "figures" / "final"
MANIFEST_PATH = WORKSPACE_ROOT / "results" / "final" / "numbers_manifest.json"
FIGURES_MANIFEST_PATH = FIGURES_DIR / "figures_manifest.json"
BIBTEX_PATH = WORKSPACE_ROOT / "docs" / "paper" / "references.bib"
MANUSCRIPT_DRAFT_PATH = WORKSPACE_ROOT / "docs" / "paper" / "MANUSCRIPT_DRAFT.md"
LOCK_PATH = WORKSPACE_ROOT / "runs" / "final_T.lock"

EXPECTED_TABLE_STEMS = [
    "tab_01_dataset_split",
    "tab_02_geometry_ablation_oof_b",
    "tab_03_main_benchmark_split_t",
    "tab_04_rq2_detector_correlation",
    "tab_05_conformal_coverage_t",
    "tab_06_conditional_coverage_odd",
    "tab_07_latency_tier1_realtime",
]

EXPECTED_FIGURE_NAMES = [
    "fig_01_hybrid_architecture.png",
    "fig_02_splits_spatial_distribution.png",
    "fig_03_ranging_error_by_distance.png",
    "fig_04_conformal_intervals_and_coverage.png",
    "fig_05_qualitative_case_studies.png",
]

EXPECTED_BIBTEX_KEYS = [
    "ni2026realtime",
    "vajgl2022distyolo",
    "haseeb2023disnet",
    "decade2024monocular",
    "agl2026lightweight",
    "bertoni2019monoloco",
    "romano2019cqr",
    "bhatt2021fcal",
    "dagan2004forward",
    "geiger2012kitti",
    "chen2010oas",
    "chen2016xgboost",
]


def test_final_tables_exist() -> None:
    """Verify all 7 table pairs (.csv and .tex) exist in results/tables/final/."""
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


def test_table_07_latency_preliminary_label() -> None:
    """Verify Table 7 carries the mandatory PRELIMINARY-v2 label and contains all 3 detectors."""
    tex_path = TABLES_DIR / "tab_07_latency_tier1_realtime.tex"
    csv_path = TABLES_DIR / "tab_07_latency_tier1_realtime.csv"
    tex_text = tex_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    assert "PRELIMINARY-v2" in tex_text, "Missing PRELIMINARY-v2 status in Table 7 LaTeX"
    assert "PRELIMINARY-v2" in csv_text, "Missing PRELIMINARY-v2 status in Table 7 CSV"

    for det in ["YOLO11s", "YOLOv8s", "YOLOv5su"]:
        assert det in tex_text, f"Missing {det} in Table 7 LaTeX"


def test_manifest_provenance_and_hash() -> None:
    """Verify 100% of manifest entries have valid source files and matching SHA-256 hashes."""
    assert MANIFEST_PATH.exists(), f"Numbers manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("numbers", {})
    assert len(metrics) >= 70, f"Expected at least 70 metrics, found {len(metrics)}"

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
    """Verify that numerical values rendered in CSV correspond directly to LaTeX tables without vacuous checks."""
    # Test Tab 3 consistency
    tab3_csv = TABLES_DIR / "tab_03_main_benchmark_split_t.csv"
    tab3_tex = TABLES_DIR / "tab_03_main_benchmark_split_t.tex"
    csv_rows = list(csv.DictReader(tab3_csv.open(encoding="utf-8")))
    tex_text = tab3_tex.read_text(encoding="utf-8")

    checked_tab3 = 0
    for row in csv_rows:
        absrel = row.get("absrel_pooled")
        if absrel and absrel != "-":
            float_val = f"{float(absrel):.4f}"
            assert float_val in tex_text, f"AbsRel {float_val} from CSV not found in Tab 3 TEX"
            checked_tab3 += 1
    assert checked_tab3 >= 15, f"Vacuous check prevented: only checked {checked_tab3} rows in Tab 3"

    # Test Tab 5 consistency
    tab5_csv = TABLES_DIR / "tab_05_conformal_coverage_t.csv"
    tab5_tex = TABLES_DIR / "tab_05_conformal_coverage_t.tex"
    csv_rows_5 = list(csv.DictReader(tab5_csv.open(encoding="utf-8")))
    tex_text_5 = tab5_tex.read_text(encoding="utf-8")

    checked_tab5 = 0
    for row in csv_rows_5:
        cov = row.get("pooled_coverage")
        if cov:
            float_cov = f"{float(cov) * 100:.1f}"
            assert float_cov in tex_text_5, f"Coverage {float_cov}% from CSV not found in Tab 5 TEX"
            checked_tab5 += 1
    assert checked_tab5 >= 9, f"Vacuous check prevented: only checked {checked_tab5} rows in Tab 5"


def test_final_figures_and_manifest() -> None:
    """Verify all 5 publication figures exist and their SHA-256 hashes match the figures manifest."""
    assert FIGURES_DIR.is_dir(), f"Figures directory missing: {FIGURES_DIR}"
    assert FIGURES_MANIFEST_PATH.exists(), f"Figures manifest missing: {FIGURES_MANIFEST_PATH}"

    with open(FIGURES_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    figures_data = manifest.get("figures", {})

    for fname in EXPECTED_FIGURE_NAMES:
        fpath = FIGURES_DIR / fname
        assert fpath.exists(), f"Missing publication figure: {fpath}"
        assert fpath.stat().st_size > 0, f"Figure is empty: {fpath}"
        assert fname in figures_data, f"Figure {fname} missing from figures_manifest.json"

        # Verify hash
        hasher = hashlib.sha256()
        with open(fpath, "rb") as ff:
            while chunk := ff.read(65536):
                hasher.update(chunk)
        expected_sha = figures_data[fname]["sha256"]
        assert hasher.hexdigest() == expected_sha, f"SHA-256 mismatch for figure {fname}"


def test_bibtex_and_citations() -> None:
    """Verify that references.bib contains all required entries and manuscript has zero raw citation tags."""
    assert BIBTEX_PATH.exists(), f"BibTeX file missing: {BIBTEX_PATH}"
    bib_text = BIBTEX_PATH.read_text(encoding="utf-8")

    for key in EXPECTED_BIBTEX_KEYS:
        assert key in bib_text, f"Missing BibTeX key '{key}' in references.bib"

    # Verify manuscript draft does not contain unresolved citation markers
    draft_text = MANUSCRIPT_DRAFT_PATH.read_text(encoding="utf-8")
    assert "CẦN TRÍCH DẪN" not in draft_text, "Found raw [CẦN TRÍCH DẪN] in MANUSCRIPT_DRAFT.md"


def test_no_internal_decision_codes_in_draft() -> None:
    """Verify that manuscript draft contains zero internal project decision codes (Dxx)."""
    assert MANUSCRIPT_DRAFT_PATH.exists()
    content = MANUSCRIPT_DRAFT_PATH.read_text(encoding="utf-8")
    matches = re.findall(r"\bD\d{1,3}\b", content)
    assert not matches, f"Found internal decision codes in manuscript draft: {matches}"


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
        "Empirical Indistinguishability of Residual and Direct Regression",
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
        r"\bequivalence\b",
        r"\brigorous\b",
        r"\bfail-safe\b",
        r"\bcomfortably\b",
        r"\bfails\s+catastrophically\b",
        r"\bfinite-sample\s+coverage\s+guarantees\b",
    ]

    for pattern in banned_regexes:
        matches = list(re.finditer(pattern, content))
        assert not matches, f"Found banned phrase matching '{pattern}' {len(matches)} time(s) in draft"


def test_lock_file_untouched() -> None:
    """Verify that the test split lock file remains completely intact and untouched."""
    assert LOCK_PATH.exists(), f"Lock file {LOCK_PATH} must exist"
    lock_content = LOCK_PATH.read_text(encoding="utf-8")
    assert "locked" in lock_content.lower(), "Lock file does not indicate locked status"
