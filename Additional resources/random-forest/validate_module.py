"""Validate that the committed notebook is rendered and error-free."""
import json
from pathlib import Path

notebook_path = Path(__file__).parent / "notebooks" / "random-forest-masterclass.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
assert code_cells, "Notebook contains no code cells"
assert all(cell.get("execution_count") is not None for cell in code_cells), "Unexecuted code cell found"
errors = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
assert not errors, errors
rendered_plots = sum(
    "image/svg+xml" in output.get("data", {}) or "image/png" in output.get("data", {})
    for cell in code_cells
    for output in cell.get("outputs", [])
)
assert rendered_plots >= 3, "Expected at least three rendered diagnostic plots"
print({"executed_code_cells": len(code_cells), "error_outputs": len(errors), "rendered_plots": rendered_plots, "status": "passed"})
