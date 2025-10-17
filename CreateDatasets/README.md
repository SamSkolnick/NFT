## Research Dataset Builder

The `web_parser.py` module assembles Kaggle-style datasets from machine learning research papers. It automates:
- Discovering ML-oriented papers via Semantic Scholar queries.
- Parsing titles/abstracts to extract referenced datasets, code repositories, and reported metrics.
- Tagging papers by discipline/task and persisting JSON + Markdown summaries.

### Usage

```python
from pathlib import Path
from CreateDatasets.web_parser import ResearchDatasetBuilder

builder = ResearchDatasetBuilder(output_dir=Path("research_datasets"))
records = builder.build(
    queries=[
        "machine learning climate science",
        "deep learning medical imaging",
        "reinforcement learning robotics",
        "natural language processing education",
    ],
    min_papers=50,
    discipline_tags={
        "Climate": ["climate", "weather", "meteorology"],
        "Healthcare": ["medical", "health", "clinical", "diagnosis"],
        "Robotics": ["robot", "control"],
        "Education": ["education", "student", "learning outcomes"],
    },
)
```

### Output

The builder writes:
- `index.json` – quick lookup across collected papers.
- `metadata/<paper_id>.json` – structured record with datasets/code/results.
- `metadata/<paper_id>.md` – human-readable summary with abstract and pointers.
- `data/` – reserved for future raw artifact downloads.

LLM fallback is enabled by default to fill missing datasets/code/results/summary using a structured JSON prompt. To disable: `ResearchDatasetBuilder(..., use_llm_fallback=False)`. Set `OPENROUTER_API_KEY` in your environment for LLM calls.

Optionally download referenced datasets (best-effort, capped size):

```python
records = builder.build(
    queries=[...],
    min_papers=50,
    download_assets=True,   # saves files under research_datasets/data/<paper_slug>/
)
```

Set `GITHUB_TOKEN` to enrich repository details; manage Semantic Scholar rate limits externally. Network access is required for live scraping; you can test offline by mocking the client classes.
