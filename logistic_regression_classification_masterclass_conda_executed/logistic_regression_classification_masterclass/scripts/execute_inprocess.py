"""Execute a notebook in one IPython process and preserve rich outputs.

This is used for build-time validation in restricted environments where a
separate Jupyter kernel cannot open its ZeroMQ sockets. Learners can use the
normal ``reexecute_notebook.py`` script inside the supplied Conda environment.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
from pathlib import Path

import nbformat
from IPython.core.displaypub import DisplayPublisher
from IPython.core.interactiveshell import InteractiveShell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logistic_regression_classification_masterclass_clean.ipynb"
DEFAULT_OUTPUT = ROOT / "logistic_regression_classification_masterclass.ipynb"


class CapturePublisher(DisplayPublisher):
    """Store rich display payloads emitted by IPython and Matplotlib."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    def publish(self, data, metadata=None, source="", transient=None, update=False, **kwargs):  # noqa: D401
        self.payloads.append(
            {
                "data": dict(data or {}),
                "metadata": dict(metadata or {}),
                "transient": dict(transient or {}),
                "update": bool(update),
            }
        )


def execute(source: Path, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/logistic-regression-mpl")
    os.environ.setdefault("IPYTHONDIR", "/tmp/logistic-regression-ipython")

    # Inline Matplotlib publishes image/png payloads through the IPython
    # display publisher, which we capture below.
    import matplotlib

    matplotlib.use("module://matplotlib_inline.backend_inline")

    notebook = nbformat.read(source, as_version=4)
    shell = InteractiveShell.instance()
    publisher = CapturePublisher()
    previous_publisher = shell.display_pub
    shell.display_pub = publisher

    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue

            stdout = io.StringIO()
            stderr = io.StringIO()
            publisher.payloads = []
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = shell.run_cell(cell.source, store_history=True, silent=False)

            cell.execution_count = shell.execution_count
            outputs = []
            if stdout.getvalue():
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=stdout.getvalue()
                    )
                )
            if stderr.getvalue():
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stderr", text=stderr.getvalue()
                    )
                )
            for payload in publisher.payloads:
                data = payload["data"]
                metadata = payload["metadata"]
                outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=data, metadata=metadata
                    )
                )

            if result.error_before_exec is not None or result.error_in_exec is not None:
                error = result.error_in_exec or result.error_before_exec
                traceback_lines = getattr(result, "traceback", None) or []
                outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback_lines,
                    )
                )
                cell.outputs = outputs
                raise RuntimeError(
                    f"Notebook cell {cell.execution_count} failed: {type(error).__name__}: {error}\n"
                    f"{stderr.getvalue()}"
                )

            cell.outputs = outputs

            # Prevent a later cell from receiving a stale figure in headless
            # builds. Inline ``show`` has already emitted its image payload.
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                pass
    finally:
        shell.display_pub = previous_publisher

    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"In-process executed notebook written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    execute(args.source, args.output)


if __name__ == "__main__":
    main()
