from __future__ import annotations

from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
OUTPUT = ROOT / "logistic_regression_classification_masterclass_rerun.ipynb"


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3", allow_errors=False)
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbformat.write(notebook, OUTPUT)
    print(f"Executed notebook written to {OUTPUT}")


if __name__ == "__main__":
    main()
