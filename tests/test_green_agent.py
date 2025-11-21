
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GreenAgent import GreenAgent

class TestGreenAgent(unittest.TestCase):
    def setUp(self):
        self.task_config = {
            "data_path": "/tmp/data",
            "test_labels": "/tmp/labels.csv",
            "constraints": {
                "max_time_seconds": 10,
                "max_memory_mb": 100,
                "max_cpus": 1.0
            }
        }

    @patch("GreenAgent.ChromaMemory")
    def test_init(self, mock_memory):
        agent = GreenAgent(self.task_config)
        self.assertEqual(agent.constraints["max_time_seconds"], 10)
        self.assertEqual(agent.constraints["max_memory_mb"], 100)

    def test_check_constraints_pass(self):
        # Mock memory to avoid init error
        with patch("GreenAgent.ChromaMemory"):
            agent = GreenAgent(self.task_config)
            
        execution = {
            "success": True,
            "time_seconds": 5,
            "memory_used_mb": 50,
            "predictions": "/tmp/preds.csv"
        }
        
        # Mock pandas read_csv to avoid file error
        with patch("pandas.read_csv") as mock_read:
            mock_read.return_value = MagicMock(columns=["prediction"])
            # Mock _valid_format since we can't easily mock the dataframe columns attribute behavior perfectly in one line
            with patch.object(agent, "_valid_format", return_value=True):
                result = agent.check_constraints(execution, agent.constraints)
                self.assertTrue(result["passed"])
                self.assertEqual(len(result["violations"]), 0)

    def test_check_constraints_fail_time(self):
        with patch("GreenAgent.ChromaMemory"):
            agent = GreenAgent(self.task_config)
            
        execution = {
            "success": True,
            "time_seconds": 15, # Exceeds 10
            "memory_used_mb": 50,
            "predictions": "/tmp/preds.csv"
        }
        
        result = agent.check_constraints(execution, agent.constraints)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Time limit exceeded" in v for v in result["violations"]))

    @patch("GreenAgent.ResearchEvaluator")
    @patch("GreenAgent.ChromaMemory")
    def test_run_research_eval(self, mock_memory, mock_evaluator_cls):
        # Setup mock instance
        mock_eval_instance = MagicMock()
        mock_evaluator_cls.return_value = mock_eval_instance
        mock_eval_instance.evaluate_research.return_value = {"final_score": 0.8}
        
        # Set env var to trigger evaluator init
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            agent = GreenAgent(self.task_config)
            
            submission = {
                "research_artifacts": "/tmp/research",
                "code_path": "/tmp/code"
            }
            
            result = agent._run_research_eval(submission)
            
            self.assertEqual(result, {"final_score": 0.8})
            mock_eval_instance.evaluate_research.assert_called_once()

if __name__ == "__main__":
    unittest.main()
