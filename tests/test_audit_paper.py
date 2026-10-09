"""
tests/test_audit_paper.py — Kiểm thử tự động tính đúng đắn và liêm chính học thuật của bài báo.
"""
from __future__ import annotations
import pytest
from scripts.audit_paper import (
    audit_numbers,
    audit_formatting_and_tone,
    audit_references,
    audit_tables_and_figures,
    run_full_paper_audit,
)


def test_audit_numbers():
    res = audit_numbers()
    assert res["status"] == "PASS", f"Numbers audit failed: {res}"
    assert res["manifest_total_keys"] >= 100
    assert len(res["missing_core_anchors"]) == 0


def test_audit_formatting_and_tone():
    res = audit_formatting_and_tone()
    assert res["status"] == "PASS", f"Formatting audit failed: {res}"
    assert len(res["found_forbidden_terms"]) == 0
    assert len(res["found_internal_dxx_codes"]) == 0
    assert len(res["found_internal_metadata_leaks"]) == 0


def test_audit_references():
    res = audit_references()
    assert res["status"] == "PASS", f"References audit failed: {res}"
    assert res["bib_entries_count"] >= 12
    assert len(res["found_hallucinated_authors"]) == 0
    assert len(res["missing_expected_bib_entries"]) == 0


def test_audit_tables_and_figures():
    res = audit_tables_and_figures()
    assert res["status"] == "PASS", f"Tables & figures audit failed: {res}"
    assert len(res["missing_tables"]) == 0
    assert res["tab2_has_all_ablations"] is True
    assert len(res["missing_figures"]) == 0
    assert res["figures_count"] == 5


def test_full_paper_audit():
    report = run_full_paper_audit()
    assert report["overall_status"] == "PASS"
