"""Validate the committed KNN notebook as an executed teaching artifact."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "knn-masterclass.ipynb"

payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

assert code_cells, "Notebook has no code cells"
assert all(cell.get("execution_count") is not None for cell in code_cells), (
    "Notebook contains unexecuted code cells"
)
assert not any(
    output.get("output_type") == "error"
    for cell in code_cells
    for output in cell.get("outputs", [])
), "Notebook contains an error output"

required_sections = {
    "From-scratch classifier",
    "Cross-validation and hyperparameter tuning",
    "Conclusions",
}
markdown = "\n".join(
    "".join(cell.get("source", []))
    for cell in payload["cells"]
    if cell["cell_type"] == "markdown"
)
missing = sorted(section for section in required_sections if section not in markdown)
assert not missing, f"Missing required notebook sections: {missing}"

print(
    f"Validated {len(code_cells)} executed code cells, "
    "zero error outputs, and all required learning sections."
)
