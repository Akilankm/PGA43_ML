"""Validate the teaching package and write validation_report.json."""

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
    ROOT / "requirements.txt",
    ROOT / "requirements-lock.txt",
    ROOT / "instructor_notes.md",
    ROOT / "student_exercises.md",
    ROOT / "references.md",
    ROOT / "data" / "DATA_SOURCE.md",
    ROOT / "data" / "data_dictionary.csv",
    ROOT / "scripts" / "build_notebook.py",
    ROOT / "scripts" / "generate_teaching_data.py",
    ROOT / "scripts" / "reexecute_notebook.py",
    ROOT / "scripts" / "render_html.py",
]


def inspect_notebook(path: Path) -> dict:
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = []
    for index, cell in enumerate(code_cells):
        compile(cell.source, f"{path.name}:cell-{index}", "exec")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(
                    {"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")}
                )
    output_count = sum(len(cell.get("outputs", [])) for cell in code_cells)
    png_count = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    )
    return {
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells),
        "output_count": output_count,
        "png_output_count": png_count,
        "errors": errors,
        "required_sections_present": all(
            keyword in "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
            for keyword in ["Leakage", "Statsmodels", "threshold", "model card"]
        ),
    }


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required package files: {missing}")

    executed = inspect_notebook(EXECUTED)
    clean = inspect_notebook(CLEAN)
    checks = {
        "required_files_present": not missing,
        "executed_notebook_json_valid": True,
        "clean_notebook_json_valid": True,
        "executed_code_compiles": not executed["errors"],
        "executed_counts_present": executed["executed_code_cells"] == executed["code_cells"],
        "executed_has_rich_png_outputs": executed["png_output_count"] >= 10,
        "executed_has_outputs": executed["output_count"] >= executed["code_cells"],
        "clean_counts_cleared": clean["executed_code_cells"] == 0,
        "clean_outputs_cleared": clean["output_count"] == 0,
        "required_sections_present": executed["required_sections_present"],
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "executed": executed,
        "clean": clean,
    }
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Package validation failed")


if __name__ == "__main__":
    main()
