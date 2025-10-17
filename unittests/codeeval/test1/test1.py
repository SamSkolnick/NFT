from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from CodeEval import evaluate
from code_extractor import CodeExtractor

TEST_DIR = Path(__file__).resolve().parent
RES = TEST_DIR / "resources"

def main():
    io = CodeExtractor(base_dir=TEST_DIR)
    prompt = io.read_prompt(RES / "prompt.txt")
    code = io.read_code(RES / "predict-hiv-progression.ipynb")
    preds = io.read_predictions(RES / "submission.csv", col=1)
    score = evaluate(prompt, code, preds)
    print("LLM score:", score)

if __name__ == "__main__":
    main()
