#!/usr/bin/env python
"""Validate the distributed notebook package without modifying it."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "abalone_linear_regression_masterclass.ipynb"
DATASET = ROOT / "data" / "abalone.csv"
ENVIRONMENT = ROOT / "environment.yaml"
REPORT = ROOT / "validation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [NOTEBOOK, DATASET, ENVIRONMENT, REPORT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    environment = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    if environment.get("name") != "abalone-linear-regression":
        raise RuntimeError("Unexpected Conda environment name.")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    errors = []
    mime_counts: Counter[str] = Counter()
    output_count = 0
    for cell_index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            output_count += 1
            if output.get("output_type") == "error":
                errors.append({
                    "cell_index": cell_index,
                    "ename": output.get("ename"),
                    "evalue": output.get("evalue"),
                })
            mime_counts.update(output.get("data", {}).keys())

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    checks = {
        "all_code_cells_executed": all(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "no_error_outputs": not errors,
        "at_least_30_embedded_plots": mime_counts["image/png"] >= 30,
        "dataset_sha256_matches": sha256(DATASET) == report["dataset"]["sha256"],
        "notebook_cell_count_matches": len(notebook.cells) == report["notebook"]["total_cells"],
        "code_cell_count_matches": len(code_cells) == report["notebook"]["code_cells"],
    }

    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({
        "checks": checks,
        "observed": {
            "total_cells": len(notebook.cells),
            "code_cells": len(code_cells),
            "outputs": output_count,
            "mime_counts": dict(mime_counts),
            "errors": errors,
            "dataset_sha256": sha256(DATASET),
        },
    }, indent=2))

    if failed:
        raise RuntimeError(f"Package validation failed: {failed}")


if __name__ == "__main__":
    main()
