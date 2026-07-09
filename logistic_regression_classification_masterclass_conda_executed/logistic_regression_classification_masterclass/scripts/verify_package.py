from __future__ import annotations

import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "logistic_regression_classification_masterclass.ipynb"
CLEAN = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
REQUIRED = [
    EXECUTED,
    CLEAN,
    ROOT / "logistic_regression_classification_masterclass.html",
    ROOT / "environment.yaml",
    ROOT / "requirements-lock.txt",
    ROOT / "scripts" / "generate_teaching_data.py",
    ROOT / "data" / "DATA_SOURCE.md",
    ROOT / "data" / "data_dictionary.csv",
]


def inspect(path: Path) -> dict:
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = []
    for index, cell in enumerate(code):
        compile(cell.source, f"{path.name}:cell-{index}", "exec")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append({"cell": index, "ename": output.ename, "evalue": output.evalue})
    return {
        "cells": len(notebook.cells),
        "code_cells": len(code),
        "executed_code_cells": sum(cell.execution_count is not None for cell in code),
        "output_count": sum(len(cell.get("outputs", [])) for cell in code),
        "errors": errors,
    }


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required package files: {missing}")
    executed = inspect(EXECUTED)
    clean = inspect(CLEAN)
    checks = {
        "executed_notebook_present": EXECUTED.exists(),
        "clean_notebook_present": CLEAN.exists(),
        "notebook_source_is_valid_python": not executed["errors"] and not clean["errors"],
        "executed_counts_present": executed["executed_code_cells"] == executed["code_cells"],
        "clean_counts_cleared": clean["executed_code_cells"] == 0,
        "clean_outputs_cleared": clean["output_count"] == 0,
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "executed": executed, "clean": clean}
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Package validation failed")


if __name__ == "__main__":
    main()
