from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DIR = ROOT / ".bootstrap"
PARTS = sorted(BOOTSTRAP_DIR.glob("build_notebook.py.b64.part*"))

if not PARTS:
    raise RuntimeError(f"Notebook builder payload is missing from {BOOTSTRAP_DIR}")

encoded = "".join(part.read_text(encoding="utf-8").strip() for part in PARTS)
source = base64.b64decode(encoded).decode("utf-8")
namespace = {
    "__name__": "__main__",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
