from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "logistic_regression_classification_masterclass.ipynb"
CLEAN = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
HTML = ROOT / "logistic_regression_classification_masterclass.html"


def run_python(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    run_python("generate_teaching_data.py")
    run_python("build_notebook.py")
    shutil.copy2(EXECUTED, CLEAN)

    notebook = nbformat.read(EXECUTED, as_version=4)
    executor = ExecutePreprocessor(timeout=1200, kernel_name="python3", allow_errors=False)
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbformat.write(notebook, EXECUTED)

    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(notebook)
    HTML.write_text(body, encoding="utf-8")

    run_python("verify_package.py")

    checksum_file = ROOT / "CHECKSUMS.sha256"
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != checksum_file
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    )
    checksum_file.write_text(
        "".join(f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}\n" for path in files),
        encoding="utf-8",
    )

    print("Materialized and validated the complete logistic-regression package:")
    for path in [EXECUTED, CLEAN, HTML, ROOT / "validation_report.json", checksum_file]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
