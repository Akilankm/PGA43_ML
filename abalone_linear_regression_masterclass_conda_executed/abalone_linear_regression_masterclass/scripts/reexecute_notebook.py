#!/usr/bin/env python
"""Re-execute the tracked teaching notebook and validate all embedded outputs."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXPECTED_VERSIONS = {
    "python": "3.13.5",
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "matplotlib": "3.10.8",
    "scipy": "1.17.0",
    "statsmodels": "0.14.6",
    "scikit-learn": "1.8.0",
}


def runtime_versions() -> dict[str, str]:
    import platform
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn
    import statsmodels

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "scikit-learn": sklearn.__version__,
    }


def validate_versions(strict: bool) -> None:
    actual = runtime_versions()
    mismatches = {
        key: {"expected": EXPECTED_VERSIONS[key], "actual": actual[key]}
        for key in EXPECTED_VERSIONS
        if EXPECTED_VERSIONS[key] != actual[key]
    }
    if mismatches:
        message = f"Runtime version mismatch: {mismatches}"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


def execute_notebook(source: Path, output: Path, kernel_name: str) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(source.parent)}},
        record_timing=True,
        allow_errors=False,
    )

    started = time.time()
    client.execute()

    notebook.metadata.kernelspec = {
        "display_name": "Python (Abalone Linear Regression)",
        "language": "python",
        "name": "abalone-linear-regression",
    }
    nbformat.write(notebook, output)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output_item
        for cell in code_cells
        for output_item in cell.get("outputs", [])
        if output_item.get("output_type") == "error"
    ]
    mime_counts: Counter[str] = Counter()
    for cell in code_cells:
        for output_item in cell.get("outputs", []):
            mime_counts.update(output_item.get("data", {}).keys())

    if errors:
        raise RuntimeError(f"Notebook contains {len(errors)} error output(s).")
    if not all(cell.get("execution_count") is not None for cell in code_cells):
        raise RuntimeError("At least one code cell was not executed.")
    if mime_counts["image/png"] < 30:
        raise RuntimeError(
            f"Expected at least 30 embedded plots, found {mime_counts['image/png']}."
        )

    print(f"Notebook executed successfully in {time.time() - started:.2f}s")
    print(f"Code cells: {len(code_cells)}")
    print(f"Embedded PNG plots: {mime_counts['image/png']}")
    print(f"Output notebook: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("abalone_linear_regression_masterclass.ipynb"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("abalone_linear_regression_masterclass.ipynb"),
    )
    parser.add_argument(
        "--kernel",
        default="abalone-linear-regression",
        help="Installed Jupyter kernelspec name.",
    )
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Warn instead of failing when package versions differ from environment.yaml.",
    )
    args = parser.parse_args()

    validate_versions(strict=not args.allow_version_drift)
    execute_notebook(args.source.resolve(), args.output.resolve(), args.kernel)


if __name__ == "__main__":
    main()
