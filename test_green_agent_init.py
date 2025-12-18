import sys
from pathlib import Path
import os

# Add GreenAgent to path
sys.path.append(str(Path(__file__).parent / "GreenAgent"))

from GreenAgent import GreenAgent

def test_legacy_data_path():
    print("Testing legacy data_path...")
    config = {
        "data_path": "/tmp/dummy_data",
        "test_labels": "/tmp/dummy_labels.csv",
        "constraints": {}
    }
    agent = GreenAgent(config)
    assert agent.train_data_path == Path("/tmp/dummy_data/train.csv")
    assert agent.test_data_path == Path("/tmp/dummy_data/test.csv")
    print("SUCCESS")

def test_explicit_paths():
    print("Testing explicit paths...")
    config = {
        "data_path": "/tmp/unused_path",
        "train_data_path": "/tmp/explicit_train.csv",
        "test_data_path": "/tmp/explicit_test.csv",
        "test_labels": "/tmp/dummy_labels.csv",
        "constraints": {}
    }
    agent = GreenAgent(config)
    assert agent.train_data_path == Path("/tmp/explicit_train.csv")
    assert agent.test_data_path == Path("/tmp/explicit_test.csv")
    print("SUCCESS")

if __name__ == "__main__":
    try:
        test_legacy_data_path()
        test_explicit_paths()
        print("\nAll initialization tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
