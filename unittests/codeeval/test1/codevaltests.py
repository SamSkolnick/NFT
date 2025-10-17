# main.py
import csv
from CodeEval import evaluate

KAGGLE_PROMPT = """Background
This contest focuses on using the nucleotide sequence of the Reverse Transcriptase (RT) and Protease (PR)...
Evaluation
The evaluation method is the misclassification error rate...
Submission
Your submission must contain predictions for 692 patients...
"""

CODE_PATH = "my_pipeline.ipynb"

def read_source(path: str) -> str:
    if path.endswith(".ipynb"):
        try:
            import nbformat
        except ImportError:
            raise SystemExit("Please `pip install nbformat` to read notebooks.")
        nb = nbformat.read(path, as_version=4)
        chunks = []
        for cell in nb.cells:
            if cell.get("cell_type") == "code":
                code = cell.get("source", "")
                cleaned = "\n".join(
                    line for line in code.splitlines()
                    if not line.strip().startswith(("%", "!", "%%"))
                )
                chunks.append(cleaned)
        source = "\n\n".join(chunks)
        return source or ""
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

SOURCE_CODE = read_source(CODE_PATH)

PRED_FILE = "submission.csv"
preds = []
with open(PRED_FILE, newline="") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if len(row) >= 2:
            try:
                preds.append(float(row[1]))
            except ValueError:
                preds.append(float(int(row[1])))

score = evaluate(KAGGLE_PROMPT, SOURCE_CODE, preds)
print("LLM score:", score)
