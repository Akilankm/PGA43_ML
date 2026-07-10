"""Render an executed notebook to a self-contained classroom HTML file.

This renderer intentionally keeps the notebook's embedded PNG, HTML-table, and
text outputs. It avoids requiring the optional JupyterLab nbconvert templates,
which are not present in some minimal environments.
"""

from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import mistune
import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logistic_regression_classification_masterclass.ipynb"
DEFAULT_OUTPUT = ROOT / "logistic_regression_classification_masterclass.html"


CSS = """
:root { --ink:#263238; --muted:#607d8b; --accent:#2a9d8f; --panel:#f7fafb; --line:#dce6ea; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.55; background:#fff; }
main { max-width:1200px; margin:0 auto; padding:2rem 2.5rem 5rem; }
h1,h2,h3 { line-height:1.2; color:#173042; margin-top:2.2rem; }
h1 { font-size:2.25rem; border-bottom:4px solid var(--accent); padding-bottom:.7rem; }
h2 { border-bottom:1px solid var(--line); padding-bottom:.4rem; }
table { border-collapse:collapse; max-width:100%; margin:1rem 0; font-size:.92rem; }
th,td { border:1px solid var(--line); padding:.42rem .6rem; vertical-align:top; }
th { background:#edf5f5; text-align:left; }
pre { overflow:auto; background:#17212b; color:#e8f1f5; border-radius:8px; padding:1rem; font-size:.84rem; line-height:1.45; }
.code { margin:1.2rem 0 2rem; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.code-title { background:#eef3f5; color:var(--muted); padding:.45rem .8rem; font-size:.8rem; font-weight:700; letter-spacing:.03em; }
.code pre { margin:0; border-radius:0; }
.output { padding:.8rem 1rem 1rem; background:#fff; border-top:1px solid var(--line); overflow:auto; }
.output img { display:block; max-width:100%; height:auto; margin:.5rem auto; }
.stream { white-space:pre-wrap; color:#31444c; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.error { color:#9b2226; white-space:pre-wrap; }
.toc { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1rem 1.3rem; }
.meta { color:var(--muted); font-size:.9rem; }
blockquote { border-left:4px solid var(--accent); margin-left:0; padding-left:1rem; color:#40545d; }
"""


def render_output(output: dict) -> str:
    output_type = output.get("output_type")
    if output_type == "stream":
        return f'<pre class="stream">{html.escape("".join(output.get("text", [])) if isinstance(output.get("text"), list) else output.get("text", ""))}</pre>'
    if output_type == "error":
        trace = "\n".join(output.get("traceback", []))
        return f'<pre class="error">{html.escape(trace or output.get("evalue", ""))}</pre>'

    data = output.get("data", {})
    fragments = []
    if "text/html" in data:
        fragments.append(data["text/html"])
    if "image/png" in data:
        image = data["image/png"]
        if isinstance(image, bytes):
            image = base64.b64encode(image).decode("ascii")
        fragments.append(f'<img src="data:image/png;base64,{image}" alt="Notebook plot output">')
    if "text/plain" in data and "text/html" not in data and "image/png" not in data:
        plain = data["text/plain"]
        fragments.append(f'<pre>{html.escape(plain)}</pre>')
    return "\n".join(fragments)


def render(source: Path, destination: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    sections = []
    code_number = 0
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            sections.append(mistune.html(cell.source))
            continue

        code_number += 1
        body = [
            '<section class="code">',
            f'<div class="code-title">Code cell {code_number} · execution count {cell.get("execution_count")}</div>',
            f'<pre><code>{html.escape(cell.source)}</code></pre>',
        ]
        outputs = cell.get("outputs", [])
        if outputs:
            body.append('<div class="output">')
            body.extend(render_output(output) for output in outputs)
            body.append("</div>")
        body.append("</section>")
        sections.append("\n".join(body))

    title = "Logistic Regression Classification Masterclass"
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title><style>{CSS}</style></head><body><main>",
            f'<p class="meta">Self-contained rendered companion for <code>{source.name}</code></p>',
            *sections,
            "</main></body></html>",
        ]
    )
    destination.write_text(document, encoding="utf-8")
    print(f"Rendered {destination} ({len(document):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == "__main__":
    main()
