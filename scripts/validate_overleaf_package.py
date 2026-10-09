"""
Validator script for adip_overleaf_package/main.tex
"""
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def validate():
    pkg_dir = Path("adip_overleaf_package")
    tex_path = pkg_dir / "main.tex"
    bib_path = pkg_dir / "references.bib"
    
    assert tex_path.exists(), "main.tex does not exist!"
    assert bib_path.exists(), "references.bib does not exist!"
    
    content = tex_path.read_text(encoding="utf-8")
    
    # 1. Balanced environments
    begins = re.findall(r"\\begin\{([^}]+)\}", content)
    ends = re.findall(r"\\end\{([^}]+)\}", content)
    print(f"Total \\begin: {len(begins)}, Total \\end: {len(ends)}")
    assert len(begins) == len(ends), f"Mismatch in environments: {len(begins)} begins vs {len(ends)} ends"
    
    stack = []
    for line_idx, line in enumerate(content.splitlines(), 1):
        for m in re.finditer(r"\\begin\{([^}]+)\}", line):
            stack.append((m.group(1), line_idx))
        for m in re.finditer(r"\\end\{([^}]+)\}", line):
            env = m.group(1)
            if not stack:
                raise ValueError(f"Unmatched \\end{{{env}}} at line {line_idx}")
            last_env, last_line = stack.pop()
            if last_env != env:
                raise ValueError(f"Environment mismatch: expected \\end{{{last_env}}} from line {last_line}, got \\end{{{env}}} at line {line_idx}")
    if stack:
        raise ValueError(f"Unclosed environments: {stack}")
    print("✅ All LaTeX environments are properly balanced.")

    # 2. Figures existence
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)
    print(f"Found {len(figs)} figures: {figs}")
    for fig in figs:
        fig_path = pkg_dir / fig
        assert fig_path.exists(), f"Figure file does not exist: {fig_path}"
    print("✅ All referenced figures exist.")

    # 3. Bib citations
    bib_content = bib_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))
    print(f"Found {len(bib_keys)} BibTeX keys: {sorted(bib_keys)}")
    
    cites = re.findall(r"\\cite\{([^}]+)\}", content)
    all_cites = set()
    for c in cites:
        for k in c.split(","):
            all_cites.add(k.strip())
    print(f"Found {len(all_cites)} cited keys in text: {sorted(all_cites)}")
    
    missing_cites = all_cites - bib_keys
    assert not missing_cites, f"Missing citation keys in references.bib: {missing_cites}"
    print("✅ All citations match references.bib.")

    # 4. Check forbidden terms or internal codes
    forbidden = ["first work", "first paper", "statistically significant", "superior performance", "outperforms all", "fail-safe", "comfortably"]
    for fb in forbidden:
        assert fb not in content.lower(), f"Forbidden term found: {fb}"
    print("✅ Zero forbidden tone words found.")

    dxx = [m for m in re.findall(r"\b[dD]\d{1,3}\b", content) if not re.match(r"^[23][dD]$", m)]
    assert not dxx, f"Internal decision codes found: {dxx}"
    print("✅ Zero internal Dxx codes found.")

    print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY! The package is ready for Overleaf!")

if __name__ == "__main__":
    validate()
