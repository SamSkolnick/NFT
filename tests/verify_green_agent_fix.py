
import os
import sys
import pandas as pd
from pathlib import Path

# Add GreenAgent to path
sys.path.append(os.path.join(os.getcwd(), "GreenAgent"))
from GreenAgent import GreenAgent

def test_direct_evaluation():
    print("Testing direct evaluation...")
    
    # Setup dummy data
    labels_file = "/tmp/test_labels.txt"
    with open(labels_file, "w") as f:
        f.write("1\n0\n1\n1\n0\n")
    
    config = {
        "test_labels": labels_file,
    }
    
    agent = GreenAgent(config)
    
    # Direct predictions matching exactly
    submission = {
        "predictions": ["1", "0", "1", "1", "0"],
        "research_artifacts": "/Users/samuelskolnick/MLEngineer/Datasets/HeartDisease/research_report.md"
    }
    
    # Mock LLM call for research eval since we don't want to hit the API in test
    import unittest.mock as mock
    with mock.patch("LLMModule.call_openrouter_tongyi") as mock_llm:
        mock_llm.return_value = "Score: 85"
        result = agent.evaluate(submission)
    
    print(f"Result: {result}")
    assert result["performance"]["accuracy"] == 1.0
    assert result["performance"]["research_quality_score"] == 85
    assert result["execution"]["success"] is True
    print("Direct evaluation test passed!")

def test_label_loading_newline():
    print("\nTesting newline label loading...")
    labels_file = "/tmp/test_labels_newline.txt"
    with open(labels_file, "w") as f:
        f.write("A\nB\nA\n")
    
    labels = GreenAgent._load_labels(labels_file)
    print(f"Loaded labels: {list(labels)}")
    assert list(labels) == ["A", "B", "A"]
    print("Newline label loading test passed!")

def test_mismatch_length():
    print("\nTesting length mismatch handling...")
    labels_file = "/tmp/test_labels_mismatch.txt"
    with open(labels_file, "w") as f:
        f.write("1\n0\n1\n")
    
    config = {"test_labels": labels_file}
    agent = GreenAgent(config)
    
    # Predictions are longer than labels
    submission = {
        "predictions": ["1", "0", "1", "1", "1"]
    }
    
    result = agent.evaluate(submission)
    print(f"Result with mismatch: {result['performance']}")
    # Accuracy should be based on the truncated length (3)
    assert result["performance"]["accuracy"] == 1.0
    print("Mismatch length handling test passed!")

if __name__ == "__main__":
    try:
        test_direct_evaluation()
        test_label_loading_newline()
        test_mismatch_length()
        print("\nAll GreenAgent tests passed!")
    except Exception as e:
        print(f"\nTests FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
