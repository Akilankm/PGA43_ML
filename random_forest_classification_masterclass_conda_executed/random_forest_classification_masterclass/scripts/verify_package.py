from pathlib import Path
import json,sys
r=json.loads((Path(__file__).resolve().parents[1]/"validation_report.json").read_text());print(json.dumps(r,indent=2));sys.exit(0 if r["passed"] else 1)
