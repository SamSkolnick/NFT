# TASKS:
#   1. Build research evaluator - I'm thinking using LLM modle
# 
# 
# 
# 
# 
# 
# 
# 
# 
from LLMModule import call_openrouter_tongyi

def evaluate_research(input_research: dict, api_key: str) -> int:
    """
    Evaluates a proposed ML research plan using an LLM via OpenRouter.
    Returns an integer score (0–100).
    """
    
        # Extract input components
    task = input_research.get("task", "")
    plan = input_research.get("plan", "")
    resources = input_research.get("resources", "")
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

