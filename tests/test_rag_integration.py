
import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GreenAgent import GreenAgent
from ResearchEval import ResearchEvaluator

class TestRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data_path": "/tmp",
            "test_labels": "labels.csv"
        }
        # Mock environment variables
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}):
            self.agent = GreenAgent(self.config)
            
        # Mock memory
        self.agent.eval_memory = MagicMock()
        self.agent.research_evaluator = MagicMock()

    def test_rag_retrieval_and_pass(self):
        # Setup mock memory return
        mock_record = MagicMock()
        mock_record.document = json.dumps({
            "research_score": {
                "final_score": 0.85,
                "summary": "Used GNNs for protein folding."
            }
        })
        self.agent.eval_memory.query.return_value = [mock_record]
        
        # Submission data
        submission = {
            "research_artifacts": "research_dir",
            "docker_image": "img",
            "code_path": "code_dir", # Inferred in real usage, but passed here for mock
            "task": "Protein Folding" # Inferred
        }
        
        # Mock inferrence of task/code_path inside _run_research_eval if needed
        # But _run_research_eval does some logic to extract them. 
        # Let's mock the internal logic or just pass enough data.
        # The method expects submission['research_artifacts'] to be a path.
        
        with patch("builtins.open", MagicMock()), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.read_text", return_value='{"task": "Protein Folding"}'):
             
            self.agent._run_research_eval(submission)
            
        # Verify memory was queried
        self.agent.eval_memory.query.assert_called_once()
        
        # Verify context was passed to evaluator
        call_args = self.agent.research_evaluator.evaluate_research.call_args
        self.assertIsNotNone(call_args)
        kwargs = call_args[1]
        
        self.assertIn("past_context", kwargs)
        self.assertIn("Used GNNs for protein folding", kwargs["past_context"])
        self.assertIn("Score: 0.85", kwargs["past_context"])

if __name__ == "__main__":
    unittest.main()
