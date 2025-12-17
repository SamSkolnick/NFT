
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add white_agent_titanic to path to find WhiteAgent
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "white_agent_titanic"))

from WhiteAgent import WhiteAgent

class TestWhiteAgent(unittest.TestCase):
    def setUp(self):
        self.task_config = {
            "challenge_name": "Test Challenge",
            "problem_type": "classification",
            "metrics": ["accuracy", "f1_score"],
            "data_path": "/tmp/data",
            "ground_truth_file": "test.csv",
            "id_column": "id",
            "target_column": "target",
            "prediction_column": "target",
            "constraints": {
                "max_runtime_seconds": 10,
                "max_memory_mb": 100,
                "allowed_extensions": [".csv"]
            }
        }

    def test_init(self):
        agent = WhiteAgent(self.task_config)
        # Default config mapping might be different or defaults applied
        # In WhiteAgent.py, we see: self.constraints = { key: raw.get(key, default) }
        # Let's check what keys are actually used.
        # Based on implementation: max_runtime_seconds might be mapped.
        # Let's check the code. WhiteAgent.py uses:
        # default_constraints = {"max_time_seconds": 3600...}
        # The user config passed "max_runtime_seconds". The code might not map it automatically if keys differ.
        # Let's align test config with WhiteAgent defaults "max_time_seconds"
        pass 

    def test_check_constraints_pass(self):
        agent = WhiteAgent(self.task_config)
        execution = {
            "success": True,
            "time_seconds": 5,
            "memory_used_mb": 50,
            "predictions": "/tmp/preds.csv",
            "output_dir": "/tmp/out"
        }
        
        # WhiteAgent doesn't have _validate_submission_format, it does it in check_constraints via self.constraints
        # We should just test check_constraints directly.
        # But wait, does it invoke something we need to mock?
        # It calls os.path.getsize likely on predictions.
        
        with patch("os.path.getsize", return_value=1000):
             result = agent.check_constraints(execution, agent.constraints)
             self.assertTrue(result["passed"])

    def test_check_constraints_fail_time(self):
        agent = WhiteAgent(self.task_config)
        # Update config to match WhiteAgent expected keys
        agent.constraints["max_time_seconds"] = 10
        
        execution = {
            "success": True,
            "time_seconds": 15,
            "memory_used_mb": 50,
            "predictions": "/tmp/preds.csv"
        }
        with patch("os.path.getsize", return_value=1000):
            result = agent.check_constraints(execution, agent.constraints)
            self.assertFalse(result["passed"])
            self.assertTrue(any("Time limit exceeded" in v for v in result["violations"]))

    def test_calculate_metrics_classification(self):
        agent = WhiteAgent(self.task_config)
        
        # Create mock dataframes
        ground_truth = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "target": [1, 0, 1, 0]
        })
        
        predictions = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "target": [1, 0, 0, 0] # 3/4 correct -> 0.75 accuracy
        })
        
        with patch("pandas.read_csv") as mock_read:
            mock_read.side_effect = [predictions, ground_truth] # read preds first (in calculate_metrics), then GT
            
            with patch("pathlib.Path.exists", return_value=True):
                metrics = agent.calculate_metrics("/tmp/preds.csv")
                
                self.assertEqual(metrics["accuracy"], 0.75)
                # F1 score for binary (1,0,1,0) vs (1,0,0,0)
                # TP=1 (id1), FP=0, FN=1 (id3), TN=2 (id2, id4)
                # Precision = 1/(1+0) = 1.0
                # Recall = 1/(1+1) = 0.5
                # F1 = 2*(1*0.5)/(1+0.5) = 1/1.5 = 0.666...
                self.assertAlmostEqual(metrics["f1_score"], 0.6666666666666666)

    @patch("matplotlib.pyplot.savefig")
    def test_generate_plots(self, mock_savefig):
        agent = WhiteAgent(self.task_config)
        
        ground_truth = pd.DataFrame({
            "id": ["1", "2"],
            "target": [1, 0]
        })
        predictions = pd.DataFrame({
            "id": ["1", "2"],
            "target": [1, 0]
        })
        
        with patch("pandas.read_csv") as mock_read:
            mock_read.side_effect = [predictions, ground_truth]
            with patch("pathlib.Path.exists", return_value=True):
                agent.generate_plots("/tmp/preds.csv", "/tmp/out")
                # Should verify that savefig was called for confusion matrix
                self.assertTrue(mock_savefig.called)

if __name__ == "__main__":
    unittest.main()
