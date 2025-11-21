
import os
import json
import sys
import logging
from typing import Dict, Any

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from GreenAgent import GreenAgent
from ResearchEval import ResearchEvaluator

# Configure logging to be less noisy for the interactive tool
logging.basicConfig(level=logging.ERROR)

SCENARIOS = {
    "1": {
        "name": "Cross-Domain: NLP to Biology (Protein Folding)",
        "task": {
            "domain": "Biology",
            "task": "Protein Folding Prediction",
            "baseline_performance": 0.5
        },
        "research_artifacts": "research_notes.md",
        "code_path": "model.py",
        "mock_files": {
            "research_notes.md": """
# Research Notes: Protein Folding with Transformers

We propose using the attention mechanism from NLP (Vaswani et al. 2017) to model amino acid residues in a protein chain.
Just as words in a sentence attend to each other to form meaning, residues attend to each other to determine 3D structure.

## Citations
- [Vaswani et al. 2017] Attention Is All You Need
- [Jumper et al. 2021] Highly accurate protein structure prediction with AlphaFold
            """,
            "model.py": """
import torch.nn as nn

class ProteinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        # ... implementation ...
            """
        }
    },
    "2": {
        "name": "Standard: ResNet for Image Classification",
        "task": {
            "domain": "Computer Vision",
            "task": "Image Classification",
            "baseline_performance": 0.7
        },
        "research_artifacts": "research.txt",
        "code_path": "resnet.py",
        "mock_files": {
            "research.txt": """
We use a standard ResNet-50 architecture pre-trained on ImageNet.
We fine-tune the last layer for our specific dataset.
            """,
            "resnet.py": """
import torchvision.models as models
model = models.resnet50(pretrained=True)
            """
        }
    }
}

class InteractiveRunner:
    def __init__(self):
        print("Initializing Green Agent components...")
        # Dummy config for GreenAgent
        self.agent = GreenAgent({
            "data_path": "/tmp", 
            "test_labels": "dummy.csv"
        })
        
        # Check for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("WARNING: ANTHROPIC_API_KEY not set. Evaluation will fail or be mocked.")
            
    def run(self):
        while True:
            print("\n" + "="*50)
            print("GREEN AGENT INTERACTIVE TESTER")
            print("="*50)
            print("Select a scenario:")
            for key, scenario in SCENARIOS.items():
                print(f"{key}. {scenario['name']}")
            print("Q. Quit")
            
            choice = input("\nEnter choice: ").strip().upper()
            
            if choice == "Q":
                break
            
            if choice in SCENARIOS:
                self.run_scenario(SCENARIOS[choice])
            else:
                print("Invalid choice.")

    def run_scenario(self, scenario: Dict[str, Any]):
        print(f"\n--- Running Scenario: {scenario['name']} ---")
        
        # 1. Setup Mock Files
        # We need to mock the file reading in ResearchEval, or actually write temp files.
        # Writing temp files is safer and easier to manage.
        import tempfile
        import shutil
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Write mock files
            for filename, content in scenario["mock_files"].items():
                (tmp_path / filename).write_text(content)
                
            research_path = str(tmp_path)
            code_path = str(tmp_path / scenario["code_path"])
            
            # 2. RAG Retrieval
            print("\n[1/3] Querying Memory (RAG)...")
            query_text = f"{scenario['task']} {scenario['mock_files'].values()}"
            similar_memories = self.agent.eval_memory.query(query_text=query_text, n_results=3)
            
            past_context_lines = []
            for mem in similar_memories:
                try:
                    data = json.loads(mem.document)
                    score = data.get("research_score", {}).get("final_score", 0.0)
                    summary = data.get("research_score", {}).get("summary", "No summary")
                    past_context_lines.append(f"- Past Run (Score: {score:.2f}): {summary}")
                except:
                    continue
            
            past_context = "\n".join(past_context_lines) if past_context_lines else "No relevant past runs found."
            
            print(f"Retrieved Context:\n{'-'*20}\n{past_context}\n{'-'*20}")
            
            # 3. Edit Context
            edit = input("Do you want to edit this context? (y/N): ").strip().lower()
            if edit == 'y':
                print("Enter new context (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                past_context = "\n".join(lines)
                print(f"New Context:\n{'-'*20}\n{past_context}\n{'-'*20}")

            # 4. Run Evaluation
            print("\n[2/3] Running Research Evaluator...")
            
            if not self.agent.research_evaluator:
                print("Notice: ANTHROPIC_API_KEY not set. Using MOCK evaluation for demonstration.")
                results = {
                    "process_score": {"overall": 0.8},
                    "impact_score": {
                        "overall": 0.9,
                        "breakdown": {
                            "cross_domain": {
                                "score": 0.95,
                                "analogy_score": 0.9,
                                "bonus": 0.25
                            },
                            "novelty": {
                                "score": 0.8,
                                "reasoning": "Mock reasoning: Novel approach using transformers for proteins."
                            }
                        }
                    },
                    "final_score": 0.92,
                    "summary": "Mock Summary: Excellent cross-domain transfer detected."
                }
            else:
                try:
                    results = self.agent.research_evaluator.evaluate_research(
                        research_artifacts_path=research_path,
                        code_path=code_path,
                        task=scenario["task"],
                        performance=0.85, # Mock performance
                        past_context=past_context
                    )
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    results = {}

            if results:
                print("\n[3/3] Evaluation Results:")
                print(json.dumps(results, indent=2))
                
                # Highlight specific metrics
                print("\nKey Metrics:")
                print(f"Final Score: {results.get('final_score', 0):.2f}")
                
                impact = results.get('impact_score', {})
                print(f"Impact Score: {impact.get('overall', 0):.2f}")
                
                cross_domain = impact.get('breakdown', {}).get('cross_domain', {})
                print(f"Cross-Domain Score: {cross_domain.get('score', 0):.2f}")
                print(f"Analogy Score: {cross_domain.get('analogy_score', 0):.2f}")
                print(f"Bonus: {cross_domain.get('bonus', 0):.2f}")
                
                novelty = impact.get('breakdown', {}).get('novelty', {})
                print(f"Novelty Score: {novelty.get('score', 0):.2f}")
                print(f"Novelty Reasoning: {novelty.get('reasoning', 'N/A')}")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    runner = InteractiveRunner()
    runner.run()
