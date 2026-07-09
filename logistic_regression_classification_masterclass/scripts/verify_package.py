from __future__ import annotations

import hashlib
import json
from pathlib import Path
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "logistic_regression_classification_masterclass.ipynb"
CLEAN = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
DATA = ROOT / "data" / "bank_marketing_teaching.csv"
REQUIRED = [
    EXECUTED, CLEAN, ROOT / "logistic_regression_classification_masterclass.html",
    ROOT / "environment.yaml", ROOT / "requirements-lock.txt", DATA,
    ROOT / "artifacts" / "final_logistic_pipeline.joblib",
    ROOT / "artifacts" / "selected_threshold.json",
    ROOT / "artifacts" / "model_metrics.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_notebook(path: Path) -> dict:
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = []
    image_outputs = 0
    unexecuted = []
    for index, cell in enumerate(code_cells):
        if cell.execution_count is None:
            unexecuted.append(index)
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                error_outputs.append({"cell": index, "ename": output.ename, "evalue": output.evalue})
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if "image/png" in data:
                image_outputs += 1
    return {
        "cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "unexecuted_code_cells": unexecuted,
        "error_outputs": error_outputs,
        "embedded_png_outputs": image_outputs,
    }


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    executed = inspect_notebook(EXECUTED)
    clean = inspect_notebook(CLEAN)
    data = pd.read_csv(DATA)

    checks = {
        "required_files_present": not missing,
        "executed_has_no_errors": not executed["error_outputs"],
        "all_executed_code_cells_ran": not executed["unexecuted_code_cells"],
        "clean_notebook_has_no_execution_counts": len(clean["unexecuted_code_cells"]) == clean["code_cells"],
        "minimum_plot_outputs": executed["embedded_png_outputs"] >= 12,
        "dataset_rows_at_least_4000": len(data) >= 4000,
        "target_is_binary": set(data["y"].unique()) == {"yes", "no"},
        "serialized_model_nonempty": (ROOT / "artifacts" / "final_logistic_pipeline.joblib").stat().st_size > 1000,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "executed_notebook": executed,
        "clean_notebook": clean,
        "dataset": {"rows": len(data), "columns": data.shape[1], "positive_rate": float((data["y"] == "yes").mean())},
        "checksums": {str(path.relative_to(ROOT)): sha256(path) for path in REQUIRED},
    }
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Package validation failed")


if __name__ == "__main__":
    main()
