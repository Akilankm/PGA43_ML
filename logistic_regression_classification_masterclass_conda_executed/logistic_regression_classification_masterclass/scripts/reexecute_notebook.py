"""Execute the clean notebook and write the instructor notebook with outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
DEFAULT_OUTPUT = ROOT / "logistic_regression_classification_masterclass.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=1_200)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logistic-regression-mpl")
    notebook = nbformat.read(args.source, as_version=4)
    executor = ExecutePreprocessor(timeout=args.timeout, kernel_name="python3", allow_errors=False)
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Executed notebook written to {args.output}")


if __name__ == "__main__":
    main()
