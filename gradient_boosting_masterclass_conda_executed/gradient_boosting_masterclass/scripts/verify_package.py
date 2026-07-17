from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "gradient_boosting_masterclass.ipynb"
REQUIRED = [
    ROOT / "README.md",
    ROOT / "environment.yaml",
    ROOT / "data" / "dataset_manifest.json",
    NOTEBOOK,
    ROOT / "rendered" / "gradient_boosting_masterclass.html",
    ROOT / "src" / "gradient_boosting_from_scratch.py",
    ROOT / "src" / "library_pipeline.py",
    ROOT / "theory" / "technical_notes.md",
    ROOT / "exercises" / "student_exercises.md",
    ROOT / "artifacts" / "metrics_summary.csv",
]

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
figure_files = sorted((ROOT / "figures").glob("figure_*.svg"))
if len(figure_files) < 4:
    missing.append(f"figures: expected at least 4, found {len(figure_files)}")
if any(path.stat().st_size == 0 for path in figure_files):
    missing.append("figures: one or more files are empty")
nb = nbformat.read(NOTEBOOK, as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]
unexecuted = [i for i, c in enumerate(code_cells) if c.execution_count is None]
errors = []
figure_outputs = 0
for i, cell in enumerate(code_cells):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append({"code_cell": i, "ename": output.get("ename"), "evalue": output.get("evalue")})
        data = output.get("data", {})
        html = data.get("text/html", "")
        if isinstance(html, list):
            html = "".join(html)
        if (
            "image/png" in data
            or "image/svg+xml" in data
            or "../figures/figure_" in html
        ):
            figure_outputs += 1

report = {
    "total_cells": len(nb.cells),
    "markdown_cells": len(markdown_cells),
    "code_cells": len(code_cells),
    "unexecuted_code_cells": unexecuted,
    "error_outputs": errors,
    "rendered_figure_outputs": figure_outputs,
    "missing_required_files": missing,
    "html_bytes": (ROOT / "rendered" / "gradient_boosting_masterclass.html").stat().st_size if not missing else 0,
}
report["passes"] = (
    not missing and not unexecuted and not errors and
    len(code_cells) >= 10 and len(markdown_cells) >= 10 and figure_outputs >= 8
)
(ROOT / "artifacts" / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passes"]:
    raise SystemExit(1)
