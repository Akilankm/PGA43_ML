"""One-time builder for the Decision Tree from Scratch masterclass."""
from __future__ import annotations

import base64
import lzma
from pathlib import Path

payload_dir = Path(__file__).with_name("decision_tree_builder_payload_b64")
payload = "".join(
    path.read_text(encoding="utf-8").strip()
    for path in sorted(payload_dir.glob("part_*.txt"))
)
source = lzma.decompress(base64.b64decode(payload)).decode("utf-8")
exec(
    compile(source, __file__, "exec"),
    {"__name__": "__main__", "__file__": __file__},
)

# This branch-only change triggers the validated resource build after merge.
