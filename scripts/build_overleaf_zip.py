"""
Package adip_overleaf_package into ADIP_2026_Overleaf_Package.zip
Ensures files are at the root of the zip archive for seamless Overleaf import.
"""
import zipfile
from pathlib import Path

def create_zip():
    pkg_dir = Path("adip_overleaf_package")
    targets = [
        Path("ADIP_2026_Overleaf_Package.zip"),
        Path("docs/paper/ADIP_2026_Overleaf_Package.zip")
    ]
    
    files_to_pack = [
        "main.tex",
        "main.bbl",
        "spie.cls",
        "spiebib.bst",
        "references.bib",
        "README_OVERLEAF.txt",
        "figures/fig_01_hybrid_architecture.png",
        "figures/fig_02_splits_spatial_distribution.png",
        "figures/fig_03_ranging_error_by_distance.png",
        "figures/fig_04_conformal_intervals_and_coverage.png",
        "figures/fig_05_qualitative_case_studies.jpg",
        "figures/fig_05_qualitative_case_studies.png"
    ]
    
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel_path in files_to_pack:
                full_path = pkg_dir / rel_path
                assert full_path.exists(), f"File {full_path} does not exist!"
                # Store in zip with rel_path as archive name (root of zip)
                zf.write(full_path, arcname=rel_path)
        print(f"Created {target} (size: {target.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_zip()
