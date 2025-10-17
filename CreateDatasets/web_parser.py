import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

import requests

@dataclass
class DatasetAsset:
    name: str
    url: str
    description: str = ""
    size_bytes: Optional[int] = None
    format_hint: Optional[str] = None


@dataclass
class CodeArtifact:
    repository_url: str
    description: str = ""
    license: Optional[str] = None
    default_branch: str = "main"


@dataclass
class ResultRecord:
    metric: str
    value: float
    dataset_name: str
    split: Optional[str] = None
    notes: str = ""


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    publication_year: Optional[int]
    venue: Optional[str]
    url: str
    summary: str = ""
    disciplines: List[str] = field(default_factory=list)
    ml_tasks: List[str] = field(default_factory=list)
    datasets: List[DatasetAsset] = field(default_factory=list)
    code: List[CodeArtifact] = field(default_factory=list)
    results: List[ResultRecord] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [asdict(asset) for asset in self.datasets]
        payload["code"] = [asdict(artifact) for artifact in self.code]
        payload["results"] = [asdict(record) for record in self.results]
        return payload


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = ",".join(
        [
            "title",
            "abstract",
            "year",
            "authors",
            "journal",
            "publicationVenue",
            "externalIds",
            "url",
            "fieldsOfStudy",
            "isOpenAccess",
        ]
    )

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search(self, query: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": self.FIELDS,
        }
        response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class GitHubClient:
    API_URL = "https://api.github.com/repos/{owner}/{repo}"

    def __init__(self, token: Optional[str] = None, timeout: int = 20):
        self.session = requests.Session()
        self.timeout = timeout
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Accept"] = "application/vnd.github+json"

    def fetch_repository(self, repo_url: str) -> Optional[Dict[str, Any]]:
        owner_repo = self._extract_repo_path(repo_url)
        if not owner_repo:
            return None
        api_url = self.API_URL.format(owner=owner_repo[0], repo=owner_repo[1])
        resp = self.session.get(api_url, timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_repo_path(url: str) -> Optional[Tuple[str, str]]:
        match = re.search(r"github\.com/([^/]+)/([^/#?]+)", url)
        if not match:
            return None
        return match.group(1), match.group(2).rstrip(".git")


class PaperContentParser:
    CODE_KEYWORDS = (
        "code",
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "source code",
        "repository",
    )
    DATASET_KEYWORDS = (
        "dataset",
        "data set",
        "benchmark",
        "corpus",
        "collection",
    )
    RESULT_METRIC_PATTERN = re.compile(
        r"(?P<metric>[A-Za-z][A-Za-z0-9@/\- ]+)\s*(?:=|:)\s*(?P<value>\d+\.\d+|\d+)",
        re.IGNORECASE,
    )

    def __init__(self, github_client: Optional[GitHubClient] = None):
        self.github_client = github_client or GitHubClient(token=None)

    def extract_code_links(self, text: str) -> List[CodeArtifact]:
        artifacts: List[CodeArtifact] = []
        for url in self._find_urls(text):
            if "github.com" in url or "gitlab.com" in url or "bitbucket.org" in url:
                description = "Referenced in paper text"
                repo_meta = self.github_client.fetch_repository(url)
                license_name = None
                default_branch = "main"
                if repo_meta:
                    license_name = (repo_meta.get("license") or {}).get("spdx_id")
                    default_branch = repo_meta.get("default_branch") or default_branch
                artifacts.append(
                    CodeArtifact(
                        repository_url=url,
                        description=description,
                        license=license_name,
                        default_branch=default_branch,
                    )
                )
        return artifacts

    def extract_datasets(self, text: str) -> List[DatasetAsset]:
        assets: List[DatasetAsset] = []
        for sentence in self._split_sentences(text):
            if any(keyword in sentence.lower() for keyword in self.DATASET_KEYWORDS):
                urls = self._find_urls(sentence)
                assets.extend(
                    [
                        DatasetAsset(name=self._guess_dataset_name(sentence), url=url, description=sentence.strip())
                        for url in urls
                    ]
                )
        return assets

    def extract_results(self, text: str, dataset_hint: str = "") -> List[ResultRecord]:
        results: List[ResultRecord] = []
        for match in self.RESULT_METRIC_PATTERN.finditer(text):
            metric = match.group("metric").strip()
            value = float(match.group("value"))
            results.append(
                ResultRecord(
                    metric=metric,
                    value=value,
                    dataset_name=dataset_hint or "unspecified",
                )
            )
        return results

    @staticmethod
    def _find_urls(text: str) -> List[str]:
        return re.findall(r"https?://[^\s\)\]]+", text)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return re.split(r"(?<=[.!?])\s+", text)

    @staticmethod
    def _guess_dataset_name(sentence: str) -> str:
        words = sentence.strip().split()
        return " ".join(words[:5]) if words else "dataset"


class LLMExtractor:
    """LLM-backed fallback to extract datasets, code, results, and summary.

    Uses the model wired in LLMModule.call_openrouter_tongyi. Expects JSON output.
    """

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def extract(self, context: Dict[str, str]) -> Dict[str, Any]:
        from LLMModule import call_openrouter_tongyi  # local import to avoid hard dep at import time

        title = context.get("title", "")
        abstract = context.get("abstract", "")
        url = context.get("url", "")
        page_text = (context.get("page_text", "") or "")[: self.max_context_chars]

        prompt = (
            "You are an expert research assistant. Read the provided paper information "
            "(title, abstract, and optional page text) and extract a structured summary "
            "with datasets, code repositories, and reported results.\n\n"
            "Return ONLY a JSON object with this exact schema keys: \n"
            "{\n"
            "  \"summary\": string,\n"
            "  \"datasets\": [ { \"name\": string, \"url\": string, \"description\": string } ],\n"
            "  \"code\": [ { \"repository_url\": string, \"description\": string } ],\n"
            "  \"results\": [ { \"metric\": string, \"value\": number, \"dataset_name\": string, \"split\": string, \"notes\": string } ]\n"
            "}\n\n"
            "Prefer authoritative links (publisher, dataset homepages, GitHub). If unknown, use empty string.\n"
            "Keep lists small and relevant.\n\n"
            f"Title: {title}\n\n"
            f"Abstract: {abstract}\n\n"
            f"URL: {url}\n\n"
            f"Page Text (optional, may be partial):\n{page_text}\n"
        )

        raw = call_openrouter_tongyi(prompt)
        return self._parse_json_safely(raw)

    @staticmethod
    def _parse_json_safely(text: str) -> Dict[str, Any]:
        # Try direct parse first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Extract first JSON object heuristically
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {
            "summary": "",
            "datasets": [],
            "code": [],
            "results": [],
        }


class ResearchDatasetBuilder:
    def __init__(
        self,
        output_dir: Path,
        paper_client: Optional[SemanticScholarClient] = None,
        content_parser: Optional[PaperContentParser] = None,
        max_results_per_query: int = 100,
        use_llm_fallback: bool = True,
        llm_extractor: Optional[LLMExtractor] = None,
        max_download_size_bytes: int = 200 * 1024 * 1024,
    ):
        self.output_dir = Path(output_dir)
        self.paper_client = paper_client or SemanticScholarClient()
        self.content_parser = content_parser or PaperContentParser()
        self.max_results_per_query = max_results_per_query
        self.use_llm_fallback = use_llm_fallback
        self.llm_extractor = llm_extractor or LLMExtractor()
        self.max_download_size_bytes = max_download_size_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        queries: Sequence[str],
        min_papers: int = 50,
        discipline_tags: Optional[Dict[str, Sequence[str]]] = None,
        download_assets: bool = False,
    ) -> List[PaperRecord]:
        aggregated: Dict[str, PaperRecord] = {}
        for query in queries:
            fetched = self._fetch_papers_for_query(query)
            for record in fetched:
                aggregated.setdefault(record.paper_id, record)
                aggregated[record.paper_id].ml_tasks = list(
                    sorted(
                        set(aggregated[record.paper_id].ml_tasks + [query])
                    )
                )

        if len(aggregated) < min_papers:
            raise RuntimeError(
                f"Insufficient papers collected ({len(aggregated)}) for target {min_papers}. "
                "Consider broadening queries."
            )

        records = list(aggregated.values())
        if discipline_tags:
            self._apply_disciplines(records, discipline_tags)

        self._persist_records(records, download_assets=download_assets)
        return records

    def _fetch_papers_for_query(self, query: str) -> List[PaperRecord]:
        all_records: List[PaperRecord] = []
        offset = 0
        while len(all_records) < self.max_results_per_query:
            batch = self.paper_client.search(query, limit=20, offset=offset)
            papers = batch.get("data", [])
            if not papers:
                break
            for paper in papers:
                record = self._convert_paper(paper, query)
                all_records.append(record)
                if len(all_records) >= self.max_results_per_query:
                    break
            offset += len(papers)
        return all_records

    def _convert_paper(self, raw: Dict[str, Any], query: str) -> PaperRecord:
        title = raw.get('title', '') or ''
        abstract = raw.get('abstract', '') or ''
        url = raw.get('url') or ''
        metadata_text = f"{title}\n{abstract}"
        code_artifacts = self.content_parser.extract_code_links(metadata_text)
        datasets = self.content_parser.extract_datasets(metadata_text)
        results = self.content_parser.extract_results(metadata_text, dataset_hint=query)

        summary = ""

        # LLM fallback if key items are missing
        if self.use_llm_fallback and (not code_artifacts or not datasets or not results or not summary):
            try:
                page_text = self._try_fetch_page_text(url)
                llm_out = self.llm_extractor.extract(
                    {"title": title, "abstract": abstract, "url": url, "page_text": page_text}
                )
                # Merge datasets
                llm_datasets = [
                    DatasetAsset(
                        name=(d.get("name") or "dataset"),
                        url=(d.get("url") or ""),
                        description=d.get("description") or "",
                    )
                    for d in (llm_out.get("datasets") or [])
                    if isinstance(d, dict)
                ]
                datasets = self._merge_datasets(datasets, llm_datasets)
                # Merge code
                llm_code = [
                    CodeArtifact(
                        repository_url=(c.get("repository_url") or c.get("url") or ""),
                        description=c.get("description") or "",
                    )
                    for c in (llm_out.get("code") or [])
                    if isinstance(c, dict)
                ]
                code_artifacts = self._merge_code(code_artifacts, llm_code)
                # Merge results
                llm_results = []
                for r in (llm_out.get("results") or []):
                    if not isinstance(r, dict):
                        continue
                    try:
                        value = float(r.get("value"))
                    except Exception:
                        continue
                    llm_results.append(
                        ResultRecord(
                            metric=str(r.get("metric") or "metric"),
                            value=value,
                            dataset_name=str(r.get("dataset_name") or query or "unspecified"),
                            split=(r.get("split") or None),
                            notes=str(r.get("notes") or ""),
                        )
                    )
                results = self._merge_results(results, llm_results)
                # Summary
                summary = str(llm_out.get("summary") or "").strip()
            except Exception:
                # LLM fallback failed; proceed with heuristics only
                pass

        authors = [a.get("name", "") for a in raw.get("authors", []) if a]
        venue = raw.get("journal", {}).get("name") if raw.get("journal") else None
        if not venue and raw.get("publicationVenue"):
            venue = raw["publicationVenue"].get("name")
        external_ids = raw.get("externalIds") or {}
        publication_year = raw.get("year")
        paper_id = external_ids.get("DOI") or external_ids.get("CorpusId") or raw.get("paperId") or raw.get("url")
        return PaperRecord(
            paper_id=str(paper_id),
            title=title,
            abstract=abstract,
            summary=summary,
            authors=[author for author in authors if author],
            publication_year=publication_year,
            venue=venue,
            url=url,
            disciplines=raw.get("fieldsOfStudy") or [],
            ml_tasks=[query],
            datasets=datasets,
            code=code_artifacts,
            results=results,
            notes={
                "is_open_access": raw.get("isOpenAccess"),
                "source": "Semantic Scholar",
            },
        )

    def _apply_disciplines(self, records: Iterable[PaperRecord], discipline_tags: Dict[str, Sequence[str]]) -> None:
        for record in records:
            inferred: List[str] = list(record.disciplines)
            for discipline, keywords in discipline_tags.items():
                haystack = f"{record.title} {record.abstract}".lower()
                if any(keyword.lower() in haystack for keyword in keywords):
                    inferred.append(discipline)
            record.disciplines = sorted(set(inferred))

    def _persist_records(self, records: Sequence[PaperRecord], download_assets: bool = False) -> None:
        meta_dir = self.output_dir / "metadata"
        data_dir = self.output_dir / "data"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        index_payload = []
        for record in records:
            slug = self._slugify(record.paper_id)
            record_path = meta_dir / f"{slug}.json"
            summary_path = meta_dir / f"{slug}.md"
            record_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            summary_path.write_text(self._render_summary_markdown(record), encoding="utf-8")
            if download_assets:
                self._download_assets(record, data_dir / slug)
            index_payload.append(
                {
                    "paper_id": record.paper_id,
                    "title": record.title,
                    "path": str(record_path.relative_to(self.output_dir)),
                    "url": record.url,
                    "disciplines": record.disciplines,
                    "ml_tasks": record.ml_tasks,
                }
            )

        index_path = self.output_dir / "index.json"
        index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\-]+", "-", value)
        return cleaned.strip("-").lower()[:96]

    @staticmethod
    def _render_summary_markdown(record: PaperRecord) -> str:
        lines = [
            f"# {record.title}",
            "",
            f"- Paper ID: `{record.paper_id}`",
            f"- URL: {record.url or 'N/A'}",
            f"- Year: {record.publication_year or 'Unknown'}",
            f"- Venue: {record.venue or 'Unknown'}",
            f"- Authors: {', '.join(record.authors) if record.authors else 'Unknown'}",
            f"- Disciplines: {', '.join(record.disciplines) if record.disciplines else 'Uncategorized'}",
            f"- ML Tasks: {', '.join(record.ml_tasks) if record.ml_tasks else 'Unspecified'}",
            "",
            "## Summary",
            (record.summary or "Not available."),
            "",
            "## Abstract",
            record.abstract or "Not available.",
            "",
            "## Datasets",
        ]
        if record.datasets:
            for asset in record.datasets:
                lines.append(f"- [{asset.name}]({asset.url}) — {asset.description}")
        else:
            lines.append("No datasets referenced.")

        lines.extend(
            [
                "",
                "## Code",
            ]
        )
        if record.code:
            for artifact in record.code:
                extra = []
                if artifact.license:
                    extra.append(f"License: {artifact.license}")
                extra.append(f"Default branch: {artifact.default_branch}")
                lines.append(f"- [{artifact.repository_url}]({artifact.repository_url}) — " + "; ".join(extra))
        else:
            lines.append("No code repositories identified.")

        lines.extend(
            [
                "",
                "## Results",
            ]
        )
        if record.results:
            for result in record.results:
                note = f" ({result.notes})" if result.notes else ""
                split = f" [{result.split}]" if result.split else ""
                lines.append(f"- {result.dataset_name}{split}: {result.metric} = {result.value}{note}")
        else:
            lines.append("No metrics extracted.")

        return "\n".join(lines)

    # ---------- helpers ----------
    @staticmethod
    def _merge_datasets(primary: List[DatasetAsset], fallback: List[DatasetAsset]) -> List[DatasetAsset]:
        seen: Set[str] = set()
        out: List[DatasetAsset] = []
        for ds in list(primary) + list(fallback):
            key = (ds.url or ds.name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ds)
        return out

    @staticmethod
    def _merge_code(primary: List[CodeArtifact], fallback: List[CodeArtifact]) -> List[CodeArtifact]:
        seen: Set[str] = set()
        out: List[CodeArtifact] = []
        for c in list(primary) + list(fallback):
            key = (c.repository_url or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    @staticmethod
    def _merge_results(primary: List[ResultRecord], fallback: List[ResultRecord]) -> List[ResultRecord]:
        def k(r: ResultRecord) -> Tuple[str, str, float]:
            return (r.metric.strip().lower(), r.dataset_name.strip().lower(), round(float(r.value), 6))

        seen: Set[Tuple[str, str, float]] = set()
        out: List[ResultRecord] = []
        for r in list(primary) + list(fallback):
            key = k(r)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    @staticmethod
    def _try_fetch_page_text(url: str, timeout: int = 15) -> str:
        if not url:
            return ""
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok and resp.headers.get("content-type", "").startswith("text/"):
                text = resp.text
                # strip scripts/styles crudely
                text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
                text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
                return re.sub(r"\s+", " ", text)
        except Exception:
            return ""
        return ""

    def _download_assets(self, record: PaperRecord, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for asset in record.datasets:
            url = (asset.url or "").strip()
            if not url or not url.startswith("http"):
                continue
            try:
                head = requests.head(url, allow_redirects=True, timeout=15)
                size = int(head.headers.get("content-length" , "0") or 0)
                if size and size > self.max_download_size_bytes:
                    continue
                filename = self._derive_filename(url, asset)
                target = dest_dir / filename
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(target, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
            except Exception:
                # best-effort download; skip on errors
                continue

    @staticmethod
    def _derive_filename(url: str, asset: DatasetAsset) -> str:
        tail = url.split("?")[0].rstrip("/").split("/")[-1]
        if not tail:
            tail = re.sub(r"[^a-z0-9\-]+", "-", asset.name.lower())[:40] or "dataset"
        return tail
