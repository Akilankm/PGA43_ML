from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
R=Path(__file__).resolve().parents[1];B=nbformat.read(R/"random_forest_classification_masterclass.ipynb",as_version=4);H,_=HTMLExporter().from_notebook_node(B);(R/"random_forest_classification_masterclass.html").write_text(H)
