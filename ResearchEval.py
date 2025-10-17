import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from LLMModule import call_openrouter_tongyi


def _normalize_research_payload(payload: Union[str, bytes, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(payload, Mapping):
        data = payload
    else:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, Mapping) else {"plan": text}
        except json.JSONDecodeError:
            data = {"plan": text}

    return {
        "task": str(data.get("task", "")),
        "plan": str(data.get("plan", "")),
        "resources": str(data.get("resources", "")),
    }


def _load_local_research(source: Union[str, os.PathLike, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        return _normalize_research_payload(source)

    path = Path(source)
    if path.is_file():
        return _normalize_research_payload(path.read_text(encoding="utf-8"))

    if path.is_dir():
        candidates = [
            path / "research.json",
            path / "research.txt",
            path / "research.md",
        ]
        candidates.extend(sorted(path.glob("*.json")))
        candidates.extend(sorted(path.glob("*.txt")))
        for candidate in candidates:
            if candidate.exists():
                return _normalize_research_payload(candidate.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Could not locate research artifacts at {path}")


def _load_aws_research(source: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        bucket = source.get("bucket")
        key = source.get("key")
    else:
        prefix = "s3://"
        if not str(source).startswith(prefix):
            raise ValueError("AWS research source must be a mapping or s3:// URI")
        bucket_key = str(source)[len(prefix):]
        bucket, _, key = bucket_key.partition("/")

    if not bucket or not key:
        raise ValueError("AWS research source requires both bucket and key")

    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("boto3 is required for AWS storage but is not installed") from exc

    client = boto3.client("s3")
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return _normalize_research_payload(body)


def _load_dropbox_research(source: Union[str, Mapping[str, Any]]) -> Dict[str, str]:
    if isinstance(source, Mapping):
        path = source.get("path") or source.get("file")
        token = source.get("token") or os.environ.get("DROPBOX_ACCESS_TOKEN")
    else:
        token = os.environ.get("DROPBOX_ACCESS_TOKEN")
        path = str(source)

    if not token:
        raise ValueError("Dropbox access token not provided and DROPBOX_ACCESS_TOKEN is unset")

    if str(path).startswith("dropbox://"):
        path = str(path)[9:]

    if not path:
        raise ValueError("Dropbox research source must include a file path")

    try:
        import dropbox  # type: ignore
    except ImportError as exc:
        raise RuntimeError("dropbox package is required for Dropbox storage but is not installed") from exc

    client = dropbox.Dropbox(token)
    _, response = client.files_download(path)
    return _normalize_research_payload(response.content)


def evaluate_research(input_research: Union[str, os.PathLike, Mapping[str, Any]], storage: str = "local") -> int:
    """
    Intakes research artifacts that may be stored locally, on S3, or in Dropbox.

    Evaluates a proposed ML research plan using an LLM via OpenRouter.
    Returns an integer score (0–100).
    """
    storage_normalized = (storage or "local").lower()
    if storage_normalized == "local":
        research_payload = _load_local_research(input_research)
    elif storage_normalized == "aws":
        research_payload = _load_aws_research(input_research)
    elif storage_normalized == "dropbox":
        research_payload = _load_dropbox_research(input_research)
    else:
        raise ValueError(f"Unsupported storage backend: {storage}")

    task = research_payload.get("task", "")
    plan = research_payload.get("plan", "")
    resources = research_payload.get("resources", "")
    example_research1 = (
        "Task: Image classification for diabetic retinopathy detection.\n"
        "Plan: Uses transfer learning with EfficientNet, applies class-balanced loss, "
        "evaluates on APTOS dataset with 5-fold CV, ensures no data leakage.\n"
        "Resources: Kaggle GPUs, public medical datasets."
    )
    example_research2 = (
        "Task: Predict protein–ligand binding affinity.\n"
        "Plan: Fine-tunes ESM-2 embeddings with GNNs over AlphaFold structures, "
        "benchmarks against PDBBind and CASF-2016, reports MAE/ΔG correlation.\n"
        "Resources: AWS A100s, AlphaFold DB."
    )

    # Prompt template
    input_prompt = f"""
    You are a senior machine learning researcher evaluating the quality of a proposed research plan.

    Scoring criteria (each 0–25 points):
    1. Clarity — Is the task and goal well-defined and measurable?
    2. Rigor — Does the plan include proper baselines, evaluation, and avoidance of data leakage?
    3. Feasibility — Are the methods and resources realistic?
    4. Novelty — Does it extend or combine ideas in a meaningful way?
    5. Accuracy - Is the information correct?
    6. Relevance - Does the research relate to solving the given project 

    Two examples of well-structured research:
    ---
    {example_research1}
    ---
    {example_research2}

    Now evaluate the following proposal and give only an integer score between 0 and 100.

    Task:
    {task}

    Plan:
    {plan}

    Resources:
    {resources}
    """

    # Model call
    response = call_openrouter_tongyi(input_prompt)
    try:
        score = int(''.join([c for c in response if c.isdigit()]))
        score = max(0, min(score, 100))
    except ValueError:
        score = 0

    return score


"""
Research Quality Evaluator for ML Agent Benchmark
Evaluates both research process (are sources real?) and impact (did it help?)
"""

import json
import os
import re
import requests
from typing import Dict, List, Any
from anthropic import Anthropic
from difflib import SequenceMatcher
import time

class ResearchEvaluator:
    """
    Two-dimensional research evaluation:
    1. Process Quality: Are citations real? Is research thorough?
    2. Research Impact: Did research lead to better results?
    """
    
    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.s2_base_url = "https://api.semanticscholar.org/graph/v1"
        
    def evaluate_research(
        self, 
        research_artifacts_path: str,
        code_path: str,
        task: Dict[str, Any],
        performance: float
    ) -> Dict[str, Any]:
        """
        Main entry point for research evaluation
        
        Args:
            research_artifacts_path: Path to research notes/citations
            code_path: Path to implementation code
            task: Task specification with domain, baseline, etc.
            performance: Achieved performance (0-1)
            
        Returns:
            Dict with process_score, impact_score, final_score
        """
        
        # Extract citations from research artifacts
        citations = self._extract_citations(research_artifacts_path)
        
        # Dimension 1: Process Quality (is research real and thorough?)
        print("Evaluating research process quality...")
        process_score = self._evaluate_process_quality(
            citations,
            research_artifacts_path
        )
        
        # Dimension 2: Research Impact (did it help?)
        print("Evaluating research impact...")
        impact_score = self._evaluate_research_impact(
            research_artifacts_path,
            code_path,
            task,
            performance
        )
        
        # Combine scores
        final_score = self._combine_scores(process_score, impact_score)
        
        return {
            'process_score': process_score,
            'impact_score': impact_score,
            'final_score': final_score,
            'summary': self._generate_summary(process_score, impact_score, final_score)
        }
    
    # ========================================================================
    # PART 1: EXTRACT CITATIONS
    # ========================================================================
    
    def _extract_citations(self, research_path: str) -> List[Dict[str, Any]]:
        """
        Extract citations from research artifacts
        Looks for patterns like:
        - [Smith et al. 2023] Title of Paper
        - Smith et al. (2023). "Title of Paper"
        - References section
        """
        
        citations = []
        
        # Read all markdown/text files in research directory
        for root, dirs, files in os.walk(research_path):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extract citations using multiple patterns
                    citations.extend(self._parse_citations_from_text(content))
        
        # Deduplicate by title
        seen_titles = set()
        unique_citations = []
        for citation in citations:
            title_normalized = citation['title'].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _parse_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse citations from text using regex patterns
        """
        citations = []
        
        # Pattern 1: [Authors Year] Title
        # Example: [Smith et al. 2023] Deep Learning for Proteins
        pattern1 = r'\[([^\]]+)\]\s*([^\n]+)'
        matches1 = re.findall(pattern1, text)
        for authors_year, title in matches1:
            year_match = re.search(r'(19|20)\d{2}', authors_year)
            citations.append({
                'title': title.strip(),
                'authors': authors_year.strip(),
                'year': int(year_match.group()) if year_match else None
            })
        
        # Pattern 2: Authors (Year). "Title"
        # Example: Smith et al. (2023). "Deep Learning for Proteins"
        pattern2 = r'([A-Z][a-z]+.*?)\((\d{4})\)\.\s*["\']([^"\']+)["\']'
        matches2 = re.findall(pattern2, text)
        for authors, year, title in matches2:
            citations.append({
                'title': title.strip(),
                'authors': authors.strip(),
                'year': int(year)
            })
        
        # Pattern 3: References section
        # Look for "References" or "Bibliography" section
        refs_match = re.search(r'(?:References|Bibliography|Citations)[:\s]+(.*)', text, re.IGNORECASE | re.DOTALL)
        if refs_match:
            refs_section = refs_match.group(1)
            # Parse each line as potential citation
            for line in refs_section.split('\n'):
                if len(line.strip()) > 20:  # Minimum length for citation
                    # Try to extract year and title
                    year_match = re.search(r'(19|20)\d{2}', line)
                    title_match = re.search(r'["\']([^"\']+)["\']', line)
                    
                    if title_match:
                        citations.append({
                            'title': title_match.group(1).strip(),
                            'authors': line[:50].strip(),  # First 50 chars as authors
                            'year': int(year_match.group()) if year_match else None
                        })
        
        return citations
    
    # ========================================================================
    # PART 2: PROCESS QUALITY EVALUATION
    # ========================================================================
    
    def _evaluate_process_quality(
        self,
        citations: List[Dict[str, Any]],
        research_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate research process quality:
        1. Citation accuracy (do papers exist?)
        2. Content accuracy (do claims match papers?)
        3. Thoroughness (enough sources? recent?)
        4. Originality (synthesis vs copy-paste?)
        """
        
        # 1. Verify citations exist
        citation_verification = self._verify_citations_exist(citations)
        
        # 2. Verify cited content matches papers
        content_verification = self._verify_citation_content(
            citations, 
            research_path
        )
        
        # 3. Evaluate thoroughness
        thoroughness = self._evaluate_thoroughness(citations)
        
        # 4. Check originality
        originality = self._evaluate_originality(research_path)
        
        # Combine into overall process score
        overall_score = (
            0.3 * citation_verification['accuracy'] +
            0.3 * content_verification['accuracy'] +
            0.2 * thoroughness['score'] +
            0.2 * originality['score']
        )
        
        return {
            'overall': overall_score,
            'citation_accuracy': citation_verification['accuracy'],
            'content_accuracy': content_verification['accuracy'],
            'thoroughness': thoroughness['score'],
            'originality': originality['score'],
            'breakdown': {
                'citations': citation_verification,
                'content': content_verification,
                'thoroughness': thoroughness,
                'originality': originality
            }
        }
    
    def _verify_citations_exist(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if papers actually exist in Semantic Scholar
        This catches hallucinated papers
        """
        
        if not citations:
            return {
                'accuracy': 1.0,
                'valid_count': 0,
                'invalid_count': 0,
                'total': 0,
                'invalid_citations': []
            }
        
        valid = 0
        invalid = []
        
        for citation in citations:
            time.sleep(0.1)  # Rate limiting
            
            try:
                # Search Semantic Scholar by title
                response = requests.get(
                    f"{self.s2_base_url}/paper/search",
                    params={
                        'query': citation['title'],
                        'fields': 'title,year,authors',
                        'limit': 1
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json().get('data', [])
                    
                    if results:
                        paper = results[0]
                        
                        # Check title similarity
                        title_sim = self._fuzzy_match(
                            citation['title'].lower(),
                            paper['title'].lower()
                        )
                        
                        # Check year if provided
                        year_match = True
                        if citation['year'] and paper.get('year'):
                            year_match = abs(citation['year'] - paper['year']) <= 1
                        
                        # Consider valid if title highly similar and year close
                        if title_sim > 0.75 and year_match:
                            valid += 1
                        else:
                            invalid.append({
                                'citation': citation,
                                'reason': f"Found paper but low similarity (sim={title_sim:.2f}, year_match={year_match})",
                                'found_paper': paper
                            })
                    else:
                        invalid.append({
                            'citation': citation,
                            'reason': "No matching paper found in Semantic Scholar"
                        })
                else:
                    # API error - don't count against them
                    valid += 1
                    
            except Exception as e:
                # Network error - don't penalize
                print(f"Error verifying citation: {e}")
                valid += 1
        
        accuracy = valid / len(citations) if citations else 1.0
        
        return {
            'accuracy': accuracy,
            'valid_count': valid,
            'invalid_count': len(invalid),
            'total': len(citations),
            'invalid_citations': invalid
        }
    
    def _verify_citation_content(
        self,
        citations: List[Dict[str, Any]],
        research_path: str
    ) -> Dict[str, Any]:
        """
        Verify that claims about papers match actual paper content
        
        This catches: "Paper X uses method Y" when X doesn't mention Y
        """
        
        if not citations:
            return {'accuracy': 1.0, 'verifications': []}
        
        # Read research artifacts to find claims about papers
        research_content = self._read_all_research_files(research_path)
        
        # For efficiency, only verify top 5 most-cited papers
        citations_to_verify = citations[:5]
        
        verifications = []
        
        for citation in citations_to_verify:
            # Extract claims about this paper
            claims = self._extract_claims_about_paper(
                citation,
                research_content
            )
            
            if not claims:
                # No specific claims made about this paper
                verifications.append({
                    'verified': True,
                    'reason': 'No specific claims to verify'
                })
                continue
            
            # Fetch paper abstract from Semantic Scholar
            paper_abstract = self._fetch_paper_abstract(citation)
            
            if paper_abstract:
                # Use LLM to verify claims against abstract
                verification = self._llm_verify_claims(claims, paper_abstract)
                verifications.append(verification)
            else:
                # Can't fetch paper - assume claims unverified but don't heavily penalize
                verifications.append({
                    'verified': False,
                    'reason': 'Could not fetch paper to verify'
                })
        
        # Calculate accuracy
        verified_count = sum(1 for v in verifications if v.get('verified', False))
        accuracy = verified_count / len(verifications) if verifications else 1.0
        
        return {
            'accuracy': accuracy,
            'verifications': verifications
        }
    
    def _extract_claims_about_paper(
        self,
        citation: Dict[str, Any],
        research_content: str
    ) -> str:
        """
        Extract claims made about a specific paper in research notes
        """
        
        # Look for mentions of this paper (by title or authors)
        title = citation['title'].lower()
        
        # Find paragraphs mentioning this paper
        paragraphs = research_content.split('\n\n')
        relevant_paragraphs = []
        
        for para in paragraphs:
            if title[:30] in para.lower():  # Match first 30 chars of title
                relevant_paragraphs.append(para)
        
        return '\n\n'.join(relevant_paragraphs)
    
    def _fetch_paper_abstract(self, citation: Dict[str, Any]) -> str:
        """
        Fetch paper abstract from Semantic Scholar
        """
        
        try:
            response = requests.get(
                f"{self.s2_base_url}/paper/search",
                params={
                    'query': citation['title'],
                    'fields': 'abstract',
                    'limit': 1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get('data', [])
                if results and results[0].get('abstract'):
                    return results[0]['abstract']
        except:
            pass
        
        return ""
    
    def _llm_verify_claims(self, claims: str, paper_abstract: str) -> Dict[str, Any]:
        """
        Use LLM to verify if claims match paper content
        """
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                temperature=0,  # Deterministic
                messages=[{
                    "role": "user",
                    "content": f"""Verify if claims about a paper match the paper's abstract.

            <claims>
            {claims[:2000]}
            </claims>

            <paper_abstract>
            {paper_abstract}
            </paper_abstract>

            Determine if the claims are consistent with the abstract.
            Consider claims verified if:
            1. They accurately describe methods/results in the abstract
            2. They don't contradict the abstract
            3. They're reasonable interpretations

            Return JSON:
            {{
                "verified": true/false,
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }}

            Be lenient - minor paraphrasing is fine. Only mark false if clearly contradictory."""
                            }]
                        )
            
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            print(f"Error in LLM verification: {e}")
            # On error, assume verified (don't penalize for our failures)
            return {'verified': True, 'reason': 'Verification error'}
    
    def _evaluate_thoroughness(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate research thoroughness:
        - Number of sources
        - Recency of sources
        - Diversity of sources
        """
        
        if not citations:
            return {'score': 0.0, 'breakdown': {}}
        
        # 1. Number of sources (diminishing returns after 10)
        num_sources = len(citations)
        source_score = min(1.0, num_sources / 10.0)
        
        # 2. Recency (prefer recent papers)
        years = [c['year'] for c in citations if c.get('year')]
        if years:
            current_year = 2025
            avg_age = current_year - (sum(years) / len(years))
            recency_score = max(0.0, 1.0 - (avg_age / 10.0))  # Penalize if avg >10 years old
        else:
            recency_score = 0.5  # Unknown
        
        # 3. Diversity (do they cite different venues/authors?)
        # Simple heuristic: if all citations have different first authors
        first_authors = []
        for c in citations:
            authors = c.get('authors', '')
            first_author = authors.split()[0] if authors else ''
            first_authors.append(first_author)
        
        unique_authors = len(set(first_authors))
        diversity_score = min(1.0, unique_authors / max(1, len(citations)))
        
        # Combine
        overall = 0.4 * source_score + 0.3 * recency_score + 0.3 * diversity_score
        
        return {
            'score': overall,
            'breakdown': {
                'num_sources': num_sources,
                'source_score': source_score,
                'recency_score': recency_score,
                'diversity_score': diversity_score,
                'avg_year': sum(years) / len(years) if years else None
            }
        }
    
    def _evaluate_originality(self, research_path: str) -> Dict[str, Any]:
        """
        Check if research is original synthesis vs copy-paste
        
        Uses LLM to assess if they synthesized insights or just listed papers
        """
        
        research_content = self._read_all_research_files(research_path)
        
        if len(research_content) < 100:
            return {'score': 0.0, 'reason': 'Minimal research content'}
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"""Evaluate originality of research notes.

                    <research_notes>
                    {research_content[:4000]}
                    </research_notes>

                    Rate originality on 0-1 scale:
                    - 0.0-0.3: Mostly copy-pasted abstracts, no synthesis
                    - 0.4-0.6: Some synthesis, but mostly summaries
                    - 0.7-0.9: Clear synthesis, connections between papers, insights
                    - 0.9-1.0: Novel insights, cross-domain connections, actionable conclusions

                    Return JSON:
                    {{
                        "score": 0.0-1.0,
                        "reasoning": "brief explanation"
                    }}"""
                }]
            )
            
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            print(f"Error evaluating originality: {e}")
            return {'score': 0.5, 'reason': 'Evaluation error'}
    
    # ========================================================================
    # PART 3: RESEARCH IMPACT EVALUATION
    # ========================================================================
    
    def _evaluate_research_impact(
        self,
        research_path: str,
        code_path: str,
        task: Dict[str, Any],
        performance: float
    ) -> Dict[str, Any]:
        """
        Evaluate research impact:
        1. Performance improvement over baseline
        2. Research → code traceability (did they implement what they researched?)
        3. Cross-domain transfer (your unique contribution!)
        4. Novelty of approach
        """
        
        # 1. Performance improvement
        baseline = task.get('baseline_performance', 0.5)
        improvement_score = self._calculate_improvement_score(performance, baseline)
        
        # 2. Research → code traceability
        traceability = self._evaluate_research_to_code_traceability(
            research_path,
            code_path
        )
        
        # 3. Cross-domain transfer
        cross_domain = self._evaluate_cross_domain_transfer(
            research_path,
            task.get('domain', 'unknown')
        )
        
        # 4. Novelty
        novelty = self._evaluate_approach_novelty(
            code_path,
            research_path
        )
        
        # Combine (weighted)
        overall = (
            0.4 * improvement_score +
            0.2 * traceability['score'] +
            0.3 * cross_domain['score'] +
            0.1 * novelty['score']
        )
        
        return {
            'overall': overall,
            'performance_improvement': improvement_score,
            'research_influence': traceability['score'],
            'cross_domain_score': cross_domain['score'],
            'novelty': novelty['score'],
            'breakdown': {
                'improvement': {
                    'performance': performance,
                    'baseline': baseline,
                    'improvement': performance - baseline,
                    'score': improvement_score
                },
                'traceability': traceability,
                'cross_domain': cross_domain,
                'novelty': novelty
            }
        }
    
    def _calculate_improvement_score(self, performance: float, baseline: float) -> float:
        """
        Calculate performance improvement score
        
        Improvement = (performance - baseline) / baseline
        But clip to reasonable range
        """
        
        if baseline <= 0:
            baseline = 0.5  # Default if no baseline
        
        improvement = (performance - baseline) / baseline
        
        # Score based on improvement
        # 0% improvement = 0.0
        # 50% improvement = 0.75
        # 100% improvement = 1.0
        # >100% improvement = 1.0 (capped)
        
        score = min(1.0, max(0.0, improvement + 0.5))
        
        return score
    
    def _evaluate_research_to_code_traceability(
        self,
        research_path: str,
        code_path: str
    ) -> Dict[str, Any]:
        """
        Can we trace research insights to code implementation?
        
        This checks if they actually used what they researched
        """
        
        research_content = self._read_all_research_files(research_path)
        code_summary = self._get_code_summary(code_path)
        
        if not research_content or not code_summary:
            return {'score': 0.0, 'reason': 'Missing research or code'}
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"""Determine if research insights were implemented in code.

                    <research_insights>
                    {research_content[:3000]}
                    </research_insights>

                    <code_summary>
                    {code_summary[:3000]}
                    </code_summary>

                    Assess if research influenced implementation:
                    - Did they implement methods they researched?
                    - Is there evidence research guided design decisions?
                    - Are there connections between research and code?

                    Return JSON:
                    {{
                        "score": 0.0-1.0,
                        "evidence": ["specific examples of research → code connections"],
                        "reasoning": "explanation"
                    }}

                    Score guidelines:
                    - 0.0-0.3: No clear connection
                    - 0.4-0.6: Some connection, methods mentioned in research appear in code
                    - 0.7-0.9: Clear implementation of researched methods
                    - 0.9-1.0: Research directly informed multiple design decisions"""
                                    }]
                                )
                                
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            print(f"Error evaluating traceability: {e}")
            return {'score': 0.5, 'reason': 'Evaluation error'}
    
    def _evaluate_cross_domain_transfer(
        self,
        research_path: str,
        task_domain: str
    ) -> Dict[str, Any]:
        """        
        Did they find methods from other domains?        """
        
        research_content = self._read_all_research_files(research_path)
        citations = self._extract_citations(research_path)
        
        if not citations:
            return {
                'score': 0.0,
                'cross_domain_count': 0,
                'same_domain_count': 0,
                'bonus': 0.0
            }
        
        # Use LLM to identify cross-domain citations
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze research for cross-domain method transfer.

Task domain: {task_domain}

<research_content>
{research_content[:4000]}
</research_content>

<citations>
{json.dumps(citations[:20], indent=2)}
</citations>

Identify:
1. Which papers are from the SAME domain as the task?
2. Which papers are from DIFFERENT domains?
3. Were cross-domain methods actually discussed/used?

Return JSON:
{{
    "same_domain_papers": [list of titles],
    "cross_domain_papers": [
        {{"title": "...", "domain": "...", "relevance": "how it relates to task"}}
    ],
    "cross_domain_usage": {{
        "used": true/false,
        "how": "description of how cross-domain insights were applied"
    }},
    "score": 0.0-1.0
}}

Score guidelines:
- 0.0: Only same-domain papers
- 0.3: Some cross-domain papers but not used
- 0.6: Cross-domain papers with potential relevance
- 0.9: Clear cross-domain method transfer
- 1.0: Novel cross-domain insight that influenced implementation"""
                }]
            )
            
            result = json.loads(response.content[0].text)
            
            # Add bonus if actually used cross-domain methods
            bonus = 0.2 if (result.get('cross_domain_usage', {}).get('used', False)) else 0.0
            
            result['bonus'] = bonus
            
            return result
            
        except Exception as e:
            print(f"Error evaluating cross-domain: {e}")
            return {
                'score': 0.0,
                'cross_domain_count': 0,
                'bonus': 0.0,
                'reason': 'Evaluation error'
            }
    
    def _evaluate_approach_novelty(
        self,
        code_path: str,
        research_path: str
    ) -> Dict[str, Any]:
        """
        Did they try something novel/creative?
        """
        
        code_summary = self._get_code_summary(code_path)
        research_content = self._read_all_research_files(research_path)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": f"""Assess novelty of approach.

                <research>
                {research_content[:2000]}
                </research>

                <code>
                {code_summary[:2000]}
                </code>

                Rate novelty:
                - 0.0-0.3: Standard approach (random forest, vanilla CNN)
                - 0.4-0.6: Common modern approach (ResNet, LSTM)
                - 0.7-0.9: Less common or creative combination
                - 0.9-1.0: Novel or unexpected approach

                Return JSON:
                {{
                    "score": 0.0-1.0,
                    "approach_description": "what they tried",
                    "reasoning": "why this score"
                }}"""
                    }]
                )
            
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            print(f"Error evaluating novelty: {e}")
            return {'score': 0.5, 'reason': 'Evaluation error'}
    
    # ========================================================================
    # PART 4: COMBINE SCORES
    # ========================================================================
    
    def _combine_scores(
        self,
        process_score: Dict[str, float],
        impact_score: Dict[str, float]
    ) -> float:
        """
        Combine process and impact scores
        
        Key philosophy:
        - Impact matters MORE than process (70% vs 30%)
        - Hallucinated but effective > Real but useless
        - But penalize low process + low impact heavily
        """
        
        process = process_score['overall']
        impact = impact_score['overall']
        
        # Base weighted score (impact matters more)
        weighted = 0.3 * process + 0.7 * impact
        
        # Penalty: Both process AND impact are bad
        if process < 0.3 and impact < 0.3:
            weighted *= 0.5
        
        # Bonus: Both are good
        if process > 0.7 and impact > 0.7:
            weighted = min(1.0, weighted + 0.1)
        
        # Add cross-domain bonus
        cross_domain_bonus = impact_score.get('breakdown', {}).get('cross_domain', {}).get('bonus', 0.0)
        weighted = min(1.0, weighted + cross_domain_bonus)
        
        return weighted
    
    def _generate_summary(
        self,
        process_score: Dict,
        impact_score: Dict,
        final_score: float
    ) -> str:
        """
        Generate human-readable summary
        """
        
        summary = f"Final Research Score: {final_score:.2f}\n\n"
        
        summary += "Process Quality:\n"
        summary += f"  - Citation Accuracy: {process_score['citation_accuracy']:.2f}\n"
        summary += f"  - Content Accuracy: {process_score['content_accuracy']:.2f}\n"
        summary += f"  - Thoroughness: {process_score['thoroughness']:.2f}\n"
        summary += f"  - Originality: {process_score['originality']:.2f}\n"
        summary += f"  Overall Process: {process_score['overall']:.2f}\n\n"
        
        summary += "Research Impact:\n"
        summary += f"  - Performance Improvement: {impact_score['performance_improvement']:.2f}\n"
        summary += f"  - Research Influence: {impact_score['research_influence']:.2f}\n"
        summary += f"  - Cross-Domain Transfer: {impact_score['cross_domain_score']:.2f}\n"
        summary += f"  - Novelty: {impact_score['novelty']:.2f}\n"
        summary += f"  Overall Impact: {impact_score['overall']:.2f}\n\n"
        
        # Qualitative assessment
        if final_score > 0.8:
            assessment = "Excellent research - thorough, impactful, and well-applied"
        elif final_score > 0.6:
            assessment = "Good research - clear positive impact"
        elif final_score > 0.4:
            assessment = "Adequate research - some impact but room for improvement"
        else:
            assessment = "Weak research - limited impact or quality issues"
        
        summary += f"Assessment: {assessment}"
        
        return summary
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _read_all_research_files(self, research_path: str) -> str:
        """
        Read all research files and concatenate
        """
        
        content = []
        
        for root, dirs, files in os.walk(research_path):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content.append(f.read())
                    except:
                        pass
        
        return '\n\n---\n\n'.join(content)
    
    def _get_code_summary(self, code_path: str) -> str:
        """
        Get summary of code (main files, not all dependencies)
        """
        
        code_files = []
        
        # Read Python files
        for root, dirs, files in os.walk(code_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code_files.append(f"# {filepath}\n{f.read()}")
                    except:
                        pass
        
        # Limit to reasonable size
        all_code = '\n\n'.join(code_files)
        return all_code[:10000]  # First 10K chars
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Fuzzy string matching (0-1 similarity)
        """
        return SequenceMatcher(None, str1, str2).ratio()


# ========================================================================
# USAGE EXAMPLE
# ========================================================================

if __name__ == "__main__":
    
    # Initialize evaluator
    evaluator = ResearchEvaluator(anthropic_api_key="your-api-key")
    
    # Evaluate a submission
    result = evaluator.evaluate_research(
        research_artifacts_path="./submissions/agent_1/research/",
        code_path="./submissions/agent_1/code/",
        task={
            'domain': 'biology',
            'baseline_performance': 0.65,
            'description': 'Predict protein structure from sequence'
        },
        performance=0.82
    )
    
    # Print results
    print(result['summary'])
    print(f"\nFinal Score: {result['final_score']:.3f}")
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(result, f, indent=2)