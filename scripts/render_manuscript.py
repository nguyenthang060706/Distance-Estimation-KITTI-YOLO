"""
scripts/render_manuscript.py — Biên dịch bản thảo bài báo từ template placeholder bằng cách nạp dữ liệu từ numbers_manifest.json.
Bảo đảm tính đúng đắn số học 100% theo thiết kế (Correct-by-Construction).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "docs" / "paper" / "MANUSCRIPT_TEMPLATE.md"
MANIFEST_PATH = REPO_ROOT / "results" / "final" / "numbers_manifest.json"
OUTPUT_DRAFT_PATH = REPO_ROOT / "docs" / "paper" / "MANUSCRIPT_DRAFT.md"

def render_manuscript() -> str:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_PATH}. Run build_numbers_manifest.py first.")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template_content = f.read()

    numbers = manifest.get("numbers", {})

    def replace_placeholder(match: re.Match) -> str:
        tag = match.group(1).strip()
        parts = tag.split(":")
        key = parts[0]
        subfield = parts[1] if len(parts) > 1 else None

        if key not in numbers:
            raise KeyError(f"Placeholder key '{key}' not found in numbers_manifest.json!")

        entry = numbers[key]
        if subfield:
            if subfield not in entry:
                raise KeyError(f"Subfield '{subfield}' not found in manifest entry for '{key}'!")
            return str(entry[subfield])
        
        # Mặc định ưu tiên display_str hoặc display_percent
        return str(entry.get("display_str", entry.get("value", "")))

    rendered = re.sub(r"\{\{num:([^\}]+)\}\}", replace_placeholder, template_content)

    # Kiểm tra không còn placeholder sót lại
    remaining = re.findall(r"\{\{num:([^\}]+)\}\}", rendered)
    if remaining:
        raise ValueError(f"Unrendered placeholders remaining: {remaining}")

    OUTPUT_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DRAFT_PATH, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Successfully compiled manuscript to {OUTPUT_DRAFT_PATH.relative_to(REPO_ROOT)}")
    return rendered

if __name__ == "__main__":
    render_manuscript()
