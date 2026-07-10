#!/usr/bin/env python
"""Render the executed linear-regression notebook to a self-contained HTML file."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "abalone_linear_regression_masterclass.ipynb"
DEFAULT_OUTPUT = ROOT / "abalone_linear_regression_masterclass.html"


def render(source: Path, destination: Path) -> None:
    notebook = nbformat.read(source, as_version=4)

    config = Config()
    config.HTMLExporter.embed_images = True
    config.HTMLExporter.exclude_input_prompt = True
    config.HTMLExporter.exclude_output_prompt = True

    exporter = HTMLExporter(config=config)
    exporter.template_name = "lab"
    body, _ = exporter.from_notebook_node(
        notebook,
        resources={"metadata": {"path": str(source.parent)}},
    )

    if "<html" not in body.lower() or "data:image/png" not in body:
        raise RuntimeError("Rendered HTML is missing the document shell or embedded PNG outputs.")

    destination.write_text(body, encoding="utf-8")
    print(f"Rendered {destination} ({len(body):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
