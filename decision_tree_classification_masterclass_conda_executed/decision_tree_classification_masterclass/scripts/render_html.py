from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]; NB=ROOT/"decision_tree_classification_masterclass.ipynb"; HTML=ROOT/"decision_tree_classification_masterclass.html"
body,_=HTMLExporter().from_notebook_node(nbformat.read(NB,as_version=4)); HTML.write_text(body,encoding="utf-8"); print("rendered",HTML)
