from __future__ import annotations
import json, csv
from pathlib import Path
from typing import List, Optional

class CodeExtractor:
    def __init__(self, base_dir: Optional[str | Path] = None):
        self.base = Path(base_dir) if base_dir else Path.cwd()

    def read_prompt(self, path: str | Path, encoding: str = "utf-8") -> str:
        p = self._abs(path)
        return p.read_text(encoding=encoding)

    def read_code(self, path: str | Path, encoding: str = "utf-8") -> str:
        """Reads .py or .ipynb and returns a single Python source string."""
        p = self._abs(path)
        if str(p).endswith(".ipynb"):
            nb = json.loads(p.read_text(encoding=encoding))
            parts = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", "")
                    if isinstance(src, list):
                        src = "".join(src)
                    cleaned = "\n".join(
                        ln for ln in src.splitlines()
                        if not ln.strip().startswith(("%", "!", "%%"))
                    )
                    parts.append(cleaned)
            return "\n\n".join(parts) or ""
        return p.read_text(encoding=encoding)

    def read_predictions(self, path: str | Path, col: int = 1, encoding: str = "utf-8") -> List[float]:
        """Reads predictions from a CSV; defaults to column 2 (index 1)."""
        p = self._abs(path)
        preds: List[float] = []
        with p.open(newline="", encoding=encoding) as f:
            r = csv.reader(f)
            _ = next(r, None)  # header (optional)
            for row in r:
                if len(row) <= col:
                    continue
                val = row[col]
                try:
                    preds.append(float(val))
                except ValueError:
                    try:
                        preds.append(float(int(val)))
                    except Exception:
                        pass
        if not preds:
            raise RuntimeError(f"No predictions parsed from {p}")
        return preds

    def _abs(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.base / p)
