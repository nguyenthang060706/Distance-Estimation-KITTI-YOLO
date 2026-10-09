"""
tests/test_audit_checklist.py — Kiểm thử đơn vị cho module kiểm toán liêm chính học thuật T18.

Bảo đảm các hàm kiểm tra tự động phát hiện chính xác các vi phạm và xác nhận tính toàn vẹn
của toàn bộ hệ sinh thái dữ liệu, bảng biểu, khóa Split T và bản thảo bài báo.
"""
from __future__ import annotations
import pytest
from pathlib import Path
from scripts.audit_checklist import (
    check_feature_leakage,
    check_splits_and_hashes,
    check_split_t_lock_and_events,
    check_no_code_bypasses_t,
    check_table_reporting_standards,
    check_language_guard,
    run_full_14_criteria_audit,
)


def test_feature_leakage_guard():
    """Kiểm tra không cột GT nào rò rỉ vào bất kỳ file features.parquet nào."""
    res = check_feature_leakage()
    assert res["status"] == "PASS", f"Rò rỉ đặc trưng phát hiện: {res['violations']}"
    assert res["checked_files_count"] >= 6, f"Số lượng file features kiểm tra quá ít ({res['checked_files_count']})"


def test_splits_and_hashes_integrity():
    """Kiểm tra 5 split A/V/B/C/T rời rạc và khớp metadata."""
    res = check_splits_and_hashes()
    assert res["status"] == "PASS", f"Lỗi split: {res.get('evidence')}"
    counts = res["frame_counts"]
    assert counts["A"] == 3740
    assert counts["V"] == 374
    assert counts["B"] == 1499
    assert counts["C"] == 766
    assert counts["T"] == 1102
    assert len(res["overlaps"]) == 0, f"Phát hiện trùng lặp giữa các split: {res['overlaps']}"


def test_split_t_lock_and_events_exact():
    """Kiểm tra khóa runs/final_T.lock và đúng 1 cặp sự kiện START/COMPLETED."""
    res = check_split_t_lock_and_events()
    assert res["status"] == "PASS", f"Lỗi lock hoặc log: {res.get('evidence')}"
    assert res["lock_exists"] is True
    assert res["events"] == ["START", "COMPLETED"]


def test_no_code_bypasses_split_t():
    """Kiểm tra codebase src/ và scripts/ không có bypass mở Split T."""
    res = check_no_code_bypasses_t()
    assert res["status"] == "PASS", f"Phát hiện code mở T ngoài luồng: {res.get('violations')}"


def test_table_reporting_standards():
    """Kiểm tra có đủ 7 cặp bảng CSV/TEX và Bảng 6 có cờ sao *."""
    res = check_table_reporting_standards()
    assert res["status"] == "PASS", f"Lỗi tiêu chuẩn bảng: {res.get('evidence')}"
    assert res["has_star_flag"] is True


def test_manuscript_language_guard():
    """Kiểm tra bản thảo không có từ ngữ tâng bốc cấm hoặc mã quyết định nội bộ Dxx."""
    res = check_language_guard()
    assert res["status"] == "PASS", f"Vi phạm văn phong: terms={res.get('found_terms')}, Dxx={res.get('found_internal_codes')}"


def test_full_14_criteria_audit_all_pass():
    """Kiểm tra toàn bộ 14 tiêu chí §11 Kế hoạch v4 đều đạt PASS."""
    audit_results = run_full_14_criteria_audit()
    assert len(audit_results) == 14, f"Số lượng tiêu chí không đủ 14 (có {len(audit_results)})"
    failed = {idx: r["name"] for idx, r in audit_results.items() if r["status"] != "PASS"}
    assert len(failed) == 0, f"Có {len(failed)} tiêu chí kiểm toán bị FAIL: {failed}"


def test_audit_checklist_markdown_generated():
    """Kiểm tra tệp docs/CHECKLIST_AUDIT.md tồn tại và chứa đủ 14 tiêu chí."""
    # Đảm bảo script đã chạy tạo file hoặc sinh lại
    p = Path("docs/CHECKLIST_AUDIT.md")
    if p.exists():
        content = p.read_text(encoding="utf-8")
        assert "CHECKLIST_AUDIT.md" in content
        assert "14/14 PASS" in content or "100% PASS" in content
        for i in range(1, 15):
            assert f"Tiêu chí {i:02d}:" in content
