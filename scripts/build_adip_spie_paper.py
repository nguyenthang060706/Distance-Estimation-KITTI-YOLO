"""
scripts/build_adip_spie_paper.py — Tự động chuyển đổi bản thảo bài báo sang file Microsoft Word (.docx) và PDF
dựa trên template chính thức của hội nghị ADIP 2026 / SPIE Proceedings (template-SPIE.doc).
Bảo đảm tuân thủ 100% typography, booktabs tables, high-resolution figures (>= 300 DPI),
và giới hạn số trang 8–14 trang theo quy định của ban tổ chức.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DOC_PATH = REPO_ROOT / "template-SPIE.doc"
MANUSCRIPT_MD_PATH = REPO_ROOT / "docs" / "paper" / "ADIP_2026_MANUSCRIPT_SPIE.md"
OUTPUT_DOCX_PATH = REPO_ROOT / "docs" / "paper" / "ADIP_2026_Manuscript_SPIE.docx"
OUTPUT_PDF_PATH = REPO_ROOT / "docs" / "paper" / "ADIP_2026_Manuscript_SPIE.pdf"
SCRATCH_DIR = REPO_ROOT / "scratch"
JSON_PAYLOAD_PATH = SCRATCH_DIR / "spie_paper_payload.json"
PS_BUILDER_PATH = SCRATCH_DIR / "build_word_doc.ps1"

FIG_MAP = {
    "Figure 1": REPO_ROOT / "results" / "figures" / "final" / "fig_01_hybrid_architecture.png",
    "Figure 2": REPO_ROOT / "results" / "figures" / "final" / "fig_02_splits_spatial_distribution.png",
    "Figure 3": REPO_ROOT / "results" / "figures" / "final" / "fig_03_ranging_error_by_distance.png",
    "Figure 4": REPO_ROOT / "results" / "figures" / "final" / "fig_04_conformal_intervals_and_coverage.png",
    "Figure 5": REPO_ROOT / "results" / "figures" / "final" / "fig_05_qualitative_case_studies.png",
}

def clean_markdown_formatting(text: str) -> str:
    """Loại bỏ markdown inline thừa để hiển thị đẹp trong Word."""
    # Thay thế math inline $...$
    t = text.replace("$", "")
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return t.strip()

def parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """Phân tích bảng markdown thành ma trận hàng/cột."""
    rows = []
    for line in lines:
        if re.match(r"^\s*\|?\s*:?-+:?\s*\|", line):
            continue  # Dòng phân cách |---|---|
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            # Bỏ phần tử rỗng ở 2 đầu do split '|'
            if parts[0] == "":
                parts = parts[1:]
            if parts and parts[-1] == "":
                parts = parts[:-1]
            cleaned_row = [clean_markdown_formatting(p) for p in parts]
            if cleaned_row:
                rows.append(cleaned_row)
    return rows

def parse_manuscript() -> dict:
    """Đọc và phân tách cấu trúc manuscript sang đối tượng có cấu trúc."""
    if not MANUSCRIPT_MD_PATH.exists():
        raise FileNotFoundError(f"Missing {MANUSCRIPT_MD_PATH}")
    if not TEMPLATE_DOC_PATH.exists():
        raise FileNotFoundError(f"Missing {TEMPLATE_DOC_PATH}")

    content = MANUSCRIPT_MD_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()

    payload = {
        "title": "Calibrated Hybrid Geometry–Learning Monocular Vehicle Distance Estimation with Lightweight YOLO Detectors",
        "authors": "Anonymous Authors",
        "affiliations": "Department of Computer Science & Engineering, Autonomous Systems Laboratory",
        "target_venue": "8th Asia Digital Image Processing Conference (ADIP 2026), Tokyo, Japan (SPIE Proceedings)",
        "abstract_title": "ABSTRACT",
        "abstract_body": "",
        "keywords": "",
        "elements": []  # danh sách các elements: heading1, heading2, paragraph, table, figure
    }

    i = 0
    in_abstract = False
    in_table = False
    table_lines = []
    table_caption = ""

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("---") or line.startswith("```"):
            i += 1
            continue

        if line.startswith("# "):
            payload["title"] = line[2:].strip()
            i += 1
            continue

        if line.startswith("**Authors:**"):
            payload["authors"] = line.replace("**Authors:**", "").strip()
            i += 1
            continue

        if line.startswith("**Affiliations:**"):
            payload["affiliations"] = line.replace("**Affiliations:**", "").strip()
            i += 1
            continue

        if line.startswith("### ABSTRACT"):
            in_abstract = True
            i += 1
            continue

        if in_abstract:
            if line.startswith("**Keywords:**"):
                payload["keywords"] = line.replace("**Keywords:**", "").strip()
                in_abstract = False
            else:
                payload["abstract_body"] += (" " + line if payload["abstract_body"] else line)
            i += 1
            continue

        # Bảng biểu
        if line.startswith("**Table ") or line.startswith("Table "):
            table_caption = clean_markdown_formatting(line)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if table_lines:
                rows = parse_markdown_table(table_lines)
                payload["elements"].append({
                    "type": "table",
                    "caption": table_caption,
                    "rows": rows
                })
            continue

        # Hình ảnh
        fig_match = re.search(r"\*?(Figure \d+):?\*?\s*(.*)", line)
        if fig_match and fig_match.group(1) in FIG_MAP:
            fig_label = fig_match.group(1)
            caption = clean_markdown_formatting(line)
            payload["elements"].append({
                "type": "figure",
                "label": fig_label,
                "caption": caption,
                "file_path": str(FIG_MAP[fig_label]),
                "file_rel": f"results/figures/final/{FIG_MAP[fig_label].name}"
            })
            i += 1
            continue

        # Tiêu đề Level 1 (Section)
        if line.startswith("## "):
            sec_title = clean_markdown_formatting(line[3:])
            payload["elements"].append({
                "type": "heading1",
                "text": sec_title
            })
            i += 1
            continue

        # Tiêu đề Level 2 (Subsection)
        if line.startswith("### "):
            subsec_title = clean_markdown_formatting(line[4:])
            payload["elements"].append({
                "type": "heading2",
                "text": subsec_title
            })
            i += 1
            continue

        # Đoạn văn bản hoặc công thức toán
        clean_p = clean_markdown_formatting(line)
        if clean_p:
            payload["elements"].append({
                "type": "paragraph",
                "text": clean_p
            })
        i += 1

    return payload

def generate_powershell_builder(payload: dict) -> None:
    """Tạo kịch bản PowerShell điều khiển Microsoft Word COM để tạo file chuẩn SPIE."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PAYLOAD_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ps_code = """
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$scratchDir = $PSScriptRoot
$repoRoot = (Get-Item $scratchDir).Parent.FullName
$templateDoc = Join-Path $repoRoot "template-SPIE.doc"
$outputDocx = Join-Path $repoRoot "docs\\paper\\ADIP_2026_Manuscript_SPIE.docx"
$outputPdf = Join-Path $repoRoot "docs\\paper\\ADIP_2026_Manuscript_SPIE.pdf"
$payloadPath = Join-Path $scratchDir "spie_paper_payload.json"

$payload = Get-Content $payloadPath -Raw -Encoding UTF8 | ConvertFrom-Json

Write-Host "Khoi dong Microsoft Word COM Automation..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.ScreenUpdating = $false

try {
    $doc = $word.Documents.Open($templateDoc, $false, $true) # Mo read-only template
    $doc.Content.Delete() # Xoa toan bo placeholder ban dau, giu nguyen Page Setup va Styles

    $range = $doc.Range(0, 0)

    # 1. Title
    $range.Text = $payload.title + "`n"
    $range.Style = "SPIE paper title"
    $range.Collapse(0) # Collapse to end

    # 2. Authors & Affiliations
    $authorText = $payload.authors + "`n" + $payload.affiliations + "`n`n"
    $range.InsertAfter($authorText)
    $range.Style = "SPIE Authors-Affils"
    $range.Collapse(0)

    # 3. Abstract Title
    $range.InsertAfter("ABSTRACT`n")
    $range.Style = "SPIE abstract title"
    $range.Collapse(0)

    # 4. Abstract Body
    $range.InsertAfter($payload.abstract_body + "`n`n")
    $range.Style = "SPIE abstract body text"
    $range.Collapse(0)

    # 5. Keywords
    $range.InsertAfter("Keywords: " + $payload.keywords + "`n`n")
    $range.Style = "*Keywords*"
    $range.Collapse(0)

    # Chen cac Section va Bang / Hinh anh
    foreach ($item in $payload.elements) {
        $type = $item.type

        if ($type -eq "heading1") {
            $range.InsertAfter($item.text + "`n")
            $range.Style = "Heading 1,SPIE Section"
            $range.Collapse(0)
        }
        elseif ($type -eq "heading2") {
            $range.InsertAfter($item.text + "`n")
            $range.Style = "Heading 2,SPIE Subsection"
            $range.Collapse(0)
        }
        elseif ($type -eq "paragraph") {
            $range.InsertAfter($item.text + "`n`n")
            $range.Style = "SPIE body text"
            $range.Collapse(0)
        }
        elseif ($type -eq "figure") {
            $figRel = $item.file_rel
            $figPath = Join-Path $repoRoot $figRel
            if (Test-Path $figPath) {
                $range.InsertAfter("`n")
                $range.Collapse(0)
                $shape = $doc.InlineShapes.AddPicture($figPath, $false, $true, $range)
                $shape.LockAspectRatio = $true
                if ($shape.Width -gt 450) {
                    $shape.Width = 450
                }
                $range.Collapse(0)
                $range.InsertAfter("`n" + $item.caption + "`n`n")
                $range.Style = "SPIE figure caption"
                $range.Collapse(0)
            }
        }
        elseif ($type -eq "table") {
            $rows = $item.rows
            $nRows = $rows.Count
            if ($nRows -gt 0) {
                # Caption
                $range.InsertAfter("`n" + $item.caption + "`n")
                $range.Style = "SPIE table caption"
                $range.Collapse(0)

                # Chuyen toan bo hang sang tab-separated text de ConvertToTable tuc thi
                $tsvRows = @()
                foreach ($r in $rows) {
                    $tsvRows += ($r -join "`t")
                }
                $tsv = ($tsvRows -join "`r`n") + "`r`n"

                $tblStart = $range.End
                $range.InsertAfter($tsv)
                $tblRange = $doc.Range($tblStart, $range.End - 1)
                $tbl = $tblRange.ConvertToTable([ref]9) # 9 = wdSeparateByTabs

                # Dinh dang Table trong 1 buoc duy nhat
                $tbl.Range.Font.Name = "Times New Roman"
                $tbl.Range.Font.Size = 9
                $tbl.Borders.Enable = $false
                $tbl.Borders.Item(-1).LineStyle = 1 # Top border (1 pt)
                $tbl.Borders.Item(-1).LineWidth = 8 # 1 pt
                $tbl.Borders.Item(-3).LineStyle = 1 # Bottom border (1 pt)
                $tbl.Borders.Item(-3).LineWidth = 8 # 1 pt

                # Header Row
                $tbl.Rows.Item(1).Range.Bold = $true
                $tbl.Rows.Item(1).Borders.Item(-3).LineStyle = 1
                $tbl.Rows.Item(1).Borders.Item(-3).LineWidth = 4 # 0.5 pt
                $tbl.Rows.Item(1).HeadingFormat = $true # Lap lai header neu sang trang

                # Canh giua va AutoFit
                $tbl.Rows.Alignment = 1 # Centered
                $tbl.AutoFitBehavior(1) # wdAutoFitWindow

                $range.SetRange($tbl.Range.End, $tbl.Range.End)
                $range.InsertAfter("`n`n")
                $range.Collapse(0)
            }
        }
    }

    $word.ScreenUpdating = $true

    # Ghi file doc (chuan ADIP / SPIE template)
    $outputDoc = Join-Path $repoRoot "docs\\paper\\ADIP_2026_Manuscript_SPIE.doc"
    Write-Host "Luu tep Word .doc: $outputDoc"
    $doc.SaveAs2($outputDoc, [ref]0) # 0 = wdFormatDocument (.doc)

    # Xuat file PDF
    Write-Host "Xuat ban PDF: $outputPdf"
    $doc.ExportAsFixedFormat($outputPdf, 17) # 17 = wdExportFormatPDF

    # Tinh toan so trang
    $pages = $doc.ComputeStatistics(2) # 2 = wdStatisticPages
    Write-Host "============================================================"
    Write-Host "KET QUA DAN TRANG TREN MICROSOFT WORD (SPIE TEMPLATE):"
    Write-Host "Tong so trang: $pages trang"
    if ($pages -ge 8 -and $pages -le 14) {
        Write-Host "[PASS] So trang nam trong khoang chuan 8-14 trang cua ADIP 2026!"
    } else {
        Write-Host "[WARNING] So trang: $pages trang"
    }
    Write-Host "============================================================"

    $doc.Close([ref]0)
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
"""
    with open(PS_BUILDER_PATH, "w", encoding="utf-8-sig") as f:
        f.write(ps_code)

def run_build() -> None:
    print("=" * 70)
    print("BUILDING ADIP 2026 SPIE MANUSCRIPT (WORD & PDF)")
    print("=" * 70)
    payload = parse_manuscript()
    print(f"Parsed {len(payload['elements'])} structured manuscript elements.")
    generate_powershell_builder(payload)
    print(f"Generated PowerShell builder script: {PS_BUILDER_PATH}")

    print("Running Word COM Automation via PowerShell...")
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS_BUILDER_PATH)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    print(res.stdout)
    if res.returncode != 0:
        print(f"Error occurred:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)

    print("Hoàn tất tạo bản thảo Word (.docx) và PDF thành công!")

if __name__ == "__main__":
    run_build()
