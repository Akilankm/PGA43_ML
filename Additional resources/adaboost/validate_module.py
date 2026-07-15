from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "adaboost-masterclass.ipynb"
REQUIRED = [
    ROOT / "README.md",
    ROOT / "environment.yaml",
    ROOT / "references.md",
    ROOT / "theory" / "01-foundations.md",
    ROOT / "theory" / "02-mathematics-and-derivation.md",
    ROOT / "theory" / "03-data-preparation-and-mechanics.md",
    ROOT / "theory" / "04-diagnostics-evaluation-and-tuning.md",
    ROOT / "theory" / "05-comparisons-failure-modes-and-production.md",
    ROOT / "src" / "adaboost_from_scratch.py",
    ROOT / "src" / "library_pipeline.py",
    ROOT / "exercises" / "student_exercises.md",
    ROOT / "data" / "README.md",
    NOTEBOOK,
]


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        print("Missing required files:", missing)
        return 1
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    unexecuted = [cell for cell in code_cells if cell.get("execution_count") is None]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    figures = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    print(f"Executed code cells: {len(code_cells) - len(unexecuted)}/{len(code_cells)}")
    print(f"Error outputs: {len(errors)}")
    print(f"Rendered figures: {figures}")
    return int(bool(unexecuted or errors or figures < 4))


if __name__ == "__main__":
    sys.exit(main())
