from pathlib import Path
import nbformat
from nbclient import NotebookClient
R=Path(__file__).resolve().parents[1];P=R/"random_forest_classification_masterclass.ipynb";B=nbformat.read(P,as_version=4);NotebookClient(B,timeout=900,kernel_name="python3",resources={"metadata":{"path":str(R)}}).execute();nbformat.write(B,P)
