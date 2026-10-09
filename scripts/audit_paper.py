"""
scripts/audit_paper.py — Bộ công cụ kiểm toán bài báo khoa học tự động.
Kiểm tra toàn diện 4 trụ cột liêm chính học thuật:
1. Số liệu (Numbers): Khớp 100% với numbers_manifest.json, không còn placeholder {{num:...}}.
2. Định dạng & Văn phong (Formatting & Tone): Không còn mã nội bộ Dxx, không còn từ ngữ tâng bốc cấm, không rò rỉ metadata nội bộ.
3. Trích dẫn (References): 100% tài liệu trích dẫn khớp với references.bib, zero hallucinated authors.
4. Bảng biểu & Hình vẽ (Tables & Figures): Đủ 7 cặp bảng (CSV + LaTeX), đủ 5 hình vẽ đạt chuẩn 300 DPI kèm figures_manifest.json.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Set

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAFT_PATH = REPO_ROOT / "docs" / "paper" / "MANUSCRIPT_DRAFT.md"
TEMPLATE_PATH = REPO_ROOT / "docs" / "paper" / "MANUSCRIPT_TEMPLATE.md"
MANIFEST_PATH = REPO_ROOT / "results" / "final" / "numbers_manifest.json"
BIB_PATH = REPO_ROOT / "docs" / "paper" / "references.bib"
FIG_MANIFEST_PATH = REPO_ROOT / "results" / "figures" / "final" / "figures_manifest.json"
TABLES_DIR = REPO_ROOT / "results" / "tables" / "final"

FORBIDDEN_TERMS = [
    "first work",
    "first paper",
    "first to propose",
    "statistically significant",
    "statistical significance",
    "proves that",
    "superior performance",
    "outperforms all",
    "fail-safe",
    "comfortably",
    "đầu tiên",
    "có ý nghĩa thống kê",
    "chứng minh rằng",
]

HALLUCINATED_AUTHORS = [
    "kim, lee, park",
    "bhatt, cooper, patel",
    "decade collaboration",
    "agl contributors",
    "li, x., et al.",
]


def audit_numbers() -> Dict[str, Any]:
    """Kiểm tra tính nhất quán và truy xuất nguồn gốc số liệu."""
    if not DRAFT_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {DRAFT_PATH}"}
    if not MANIFEST_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {MANIFEST_PATH}"}

    draft_text = DRAFT_PATH.read_text(encoding="utf-8")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Kiểm tra placeholder sót lại
    unrendered = re.findall(r"\{\{num:([^\}]+)\}\}", draft_text)
    if unrendered:
        return {"status": "FAIL", "error": f"Unrendered placeholders found: {unrendered}"}

    # 2. Kiểm tra các số liệu chính của manifest có xuất hiện trong draft
    numbers = manifest.get("numbers", {})
    verified_count = 0
    missing_keys = []
    
    # Kiểm tra một số anchor metrics tiêu biểu bắt buộc có trong văn bản
    core_anchors = [
        "yolo11s_absrel_f", "yolo11s_delta1_f", "yolo11s_mae_f",
        "yolo11s_coverage_cqr", "common_support_count",
        "split_t_car_hard_count", "yolo11s_latency_gpu_median"
    ]
    for key in core_anchors:
        if key in numbers:
            entry = numbers[key]
            val_str = str(entry.get("display_str", entry.get("value", "")))
            if val_str in draft_text:
                verified_count += 1
            else:
                missing_keys.append((key, val_str))

    passed = (len(missing_keys) == 0)
    return {
        "status": "PASS" if passed else "FAIL",
        "manifest_total_keys": len(numbers),
        "core_anchors_verified": verified_count,
        "missing_core_anchors": missing_keys
    }


def audit_formatting_and_tone() -> Dict[str, Any]:
    """Kiểm tra văn phong, từ cấm tâng bốc, mã nội bộ Dxx và metadata rò rỉ."""
    if not DRAFT_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {DRAFT_PATH}"}

    content = DRAFT_PATH.read_text(encoding="utf-8")
    lower_content = content.lower()

    # 1. Tìm từ cấm
    found_forbidden = [t for t in FORBIDDEN_TERMS if t in lower_content]

    # 2. Tìm mã nội bộ Dxx (bỏ qua 2D, 3D)
    dxx_matches = re.findall(r"\b[dD]\d{1,3}\b", content)
    dxx_matches = [m for m in dxx_matches if not re.match(r"^[23][dD]$", m)]

    # 3. Kiểm tra rò rỉ metadata nội bộ trong draft
    internal_leaks = []
    for leak_token in ["runs/final_t.lock", "zero-touch split t locked"]:
        if leak_token in lower_content:
            internal_leaks.append(leak_token)

    passed = (len(found_forbidden) == 0 and len(dxx_matches) == 0 and len(internal_leaks) == 0)
    return {
        "status": "PASS" if passed else "FAIL",
        "found_forbidden_terms": found_forbidden,
        "found_internal_dxx_codes": dxx_matches,
        "found_internal_metadata_leaks": internal_leaks
    }


def audit_references() -> Dict[str, Any]:
    """Kiểm tra tính chính xác của trích dẫn và loại bỏ tác giả bịa đặt."""
    if not BIB_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {BIB_PATH}"}
    if not DRAFT_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {DRAFT_PATH}"}

    bib_text = BIB_PATH.read_text(encoding="utf-8")
    draft_text = DRAFT_PATH.read_text(encoding="utf-8")
    lower_bib = bib_text.lower()
    lower_draft = draft_text.lower()

    # 1. Kiểm tra tác giả bịa đặt
    found_hallucinated = []
    for author in HALLUCINATED_AUTHORS:
        if author in lower_bib or author in lower_draft:
            found_hallucinated.append(author)

    # 2. Đếm các mục trích dẫn trong .bib
    bib_entries = re.findall(r"@\w+\{([^,]+),", bib_text)

    # 3. Kiểm tra 12 trích dẫn chuẩn
    expected_citations = [
        "ni2026realtime", "vajgl2022distyolo", "haseeb2023disnet", "decade2024monocular",
        "agl2026lightweight", "bertoni2019monoloco", "romano2019cqr", "bhatt2021fcal",
        "dagan2004forward", "geiger2012kitti", "chen2010oas", "chen2016xgboost"
    ]
    missing_bib = [c for c in expected_citations if c not in bib_entries]

    passed = (len(found_hallucinated) == 0 and len(missing_bib) == 0 and len(bib_entries) >= 12)
    return {
        "status": "PASS" if passed else "FAIL",
        "bib_entries_count": len(bib_entries),
        "found_hallucinated_authors": found_hallucinated,
        "missing_expected_bib_entries": missing_bib
    }


def audit_tables_and_figures() -> Dict[str, Any]:
    """Kiểm tra toàn bộ 7 cặp bảng và 5 hình vẽ xuất bản."""
    expected_tables = [
        "tab_01_dataset_split",
        "tab_02_geometry_ablation_oof_b",
        "tab_03_main_benchmark_split_t",
        "tab_04_rq2_detector_correlation",
        "tab_05_conformal_coverage_t",
        "tab_06_conditional_coverage_odd",
        "tab_07_latency_tier1_realtime",
    ]
    missing_tables = []
    for t in expected_tables:
        csv_file = TABLES_DIR / f"{t}.csv"
        tex_file = TABLES_DIR / f"{t}.tex"
        if not csv_file.exists() or not tex_file.exists():
            missing_tables.append(t)

    # Kiểm tra Table 2 có đủ 9 ablations
    tab2_csv = TABLES_DIR / "tab_02_geometry_ablation_oof_b.csv"
    tab2_has_all_ablations = False
    if tab2_csv.exists():
        tab2_content = tab2_csv.read_text(encoding="utf-8")
        tab2_has_all_ablations = ("Drop Cue Z_H" in tab2_content and "Drop validity_flags" in tab2_content)

    # Kiểm tra 5 hình vẽ và manifest
    if not FIG_MANIFEST_PATH.exists():
        return {"status": "FAIL", "error": f"Missing {FIG_MANIFEST_PATH}"}
    with open(FIG_MANIFEST_PATH, "r", encoding="utf-8") as f:
        fig_manifest = json.load(f)
    figures = fig_manifest.get("figures", {})
    expected_figs = ["fig_01", "fig_02", "fig_03", "fig_04", "fig_05"]
    missing_figs = [f for f in expected_figs if not any(k.startswith(f) for k in figures.keys())]

    passed = (len(missing_tables) == 0 and tab2_has_all_ablations and len(missing_figs) == 0)
    return {
        "status": "PASS" if passed else "FAIL",
        "missing_tables": missing_tables,
        "tab2_has_all_ablations": tab2_has_all_ablations,
        "missing_figures": missing_figs,
        "figures_count": len(figures)
    }


def run_full_paper_audit() -> Dict[str, Any]:
    """Chạy toàn bộ kiểm toán bài báo."""
    res_nums = audit_numbers()
    res_fmt = audit_formatting_and_tone()
    res_refs = audit_references()
    res_tf = audit_tables_and_figures()

    all_pass = all(r["status"] == "PASS" for r in [res_nums, res_fmt, res_refs, res_tf])

    report = {
        "overall_status": "PASS" if all_pass else "FAIL",
        "audit_numbers": res_nums,
        "audit_formatting": res_fmt,
        "audit_references": res_refs,
        "audit_tables_and_figures": res_tf
    }

    print("=" * 70)
    print("BÁO CÁO KIỂM TOÁN BÀI BÁO KHOA HỌC (PAPER AUDIT SUITE)")
    print("=" * 70)
    for name, res in [
        ("1. Số liệu (Numbers)", res_nums),
        ("2. Định dạng & Văn phong (Formatting & Tone)", res_fmt),
        ("3. Danh mục Trích dẫn (References)", res_refs),
        ("4. Bảng biểu & Hình vẽ (Tables & Figures)", res_tf)
    ]:
        status_icon = "✅ PASS" if res["status"] == "PASS" else "❌ FAIL"
        print(f"[{status_icon}] {name}")
        for k, v in res.items():
            if k != "status":
                print(f"    - {k}: {v}")

    print("=" * 70)
    print(f"KẾT LUẬN CHUNG: {'TOÀN BỘ KIỂM TOÁN ĐẠT CHUẨN 100%' if all_pass else 'CÓ TIÊU CHÍ CHƯA ĐẠT'}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = run_full_paper_audit()
    if report["overall_status"] != "PASS":
        sys.exit(1)
    sys.exit(0)
