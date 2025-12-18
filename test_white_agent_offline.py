
import sys
import argparse
import json
from pathlib import Path

# Add SolverAgent to path
sys.path.append(str(Path(__file__).parent / "SolverAgent"))

from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Test White Agent logic offline.")
    parser.add_argument("--data", help="Path to training CSV", required=True)
    parser.add_argument("--task", help="Task description", default="Predict the target variable")
    parser.add_argument("--model", help="LLM Model (e.g., gpt-4o, gpt-3.5-turbo)", default="gpt-4o")
    
    args = parser.parse_args()
    
    print(f"Testing White Agent on {args.data} using {args.model}...")
    try:
        data_path = Path(args.data).resolve()
        if not data_path.exists():
            print(f"Error: File {data_path} does not exist.")
            return

        result = train_model(
            task_desc=args.task, 
            constraints="Simplicity", 
            llm_model=args.model, 
            data_path=data_path
        )
        
        print("\n--- Training Successful ---")
        print(f"Selected Model: {result['selected_model']}")
        print(f"Model Saved To: {result['model_path']}")
        print("Validation Report:")
        print(json.dumps(result["validation_report"], indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
