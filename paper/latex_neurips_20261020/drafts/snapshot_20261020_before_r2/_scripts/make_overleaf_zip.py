"""Package the Overleaf upload archive for the ALFNet LaTeX project.

Run from the repository root:

    python paper/latex_neurips_20261020/_scripts/make_overleaf_zip.py

Only the files needed to compile are included. The working documents (README,
change notes for the author, provenance JSON, drafts, notes, scripts and the original
template) stay in the repository and are not uploaded to Overleaf.
"""
import hashlib
import os
import zipfile

ROOT = os.getcwd()
BASE = os.path.join(ROOT, "paper", "latex_neurips_20261020")
TARGET = os.path.join(ROOT, "paper", "latex_neurips_20261020_overleaf.zip")
FILES = ["main.tex", "neurips_2026.sty", "references.bib"]
DIRS = {"sections": ".tex", "tables": ".tex", "figures": None}


def main():
    members = list(FILES)
    for d, suffix in DIRS.items():
        for name in sorted(os.listdir(os.path.join(BASE, d))):
            if suffix is None or name.endswith(suffix):
                members.append(f"{d}/{name}")
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in members:
            z.write(os.path.join(BASE, *rel.split("/")), arcname=rel)
    with open(TARGET, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print(f"{os.path.relpath(TARGET, ROOT)}: {len(members)} files, "
          f"{os.path.getsize(TARGET)} bytes, sha256 {digest}")
    for rel in members:
        print("  ", rel)


if __name__ == "__main__":
    main()
