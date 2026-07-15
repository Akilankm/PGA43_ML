from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]; NB=ROOT/"random_forest_classification_masterclass.ipynb"
nb=nbformat.read(NB,as_version=4); NotebookClient(nb,timeout=1800,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute(); nbformat.write(nb,NB); print("executed",NB)
