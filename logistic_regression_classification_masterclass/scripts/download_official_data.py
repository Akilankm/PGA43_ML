from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        members = outer.namelist()
        bank_zip_name = next(name for name in members if name.endswith("bank.zip"))
        with outer.open(bank_zip_name) as nested_file:
            nested_payload = nested_file.read()
    with zipfile.ZipFile(io.BytesIO(nested_payload)) as nested:
        csv_name = next(name for name in nested.namelist() if name.endswith("bank-full.csv"))
        content = nested.read(csv_name)
    output = OUTPUT_DIR / "bank-full.csv"
    output.write_bytes(content)
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print("The official file is semicolon-delimited. Use pd.read_csv(path, sep=';').")


if __name__ == "__main__":
    main()
