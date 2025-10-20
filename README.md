This is the planning so far

Green Agent Components:
    1. Database with Chromadb
    2. ResearchEval
    3. Run input model on test data
    4. Output results 
    

Planed workflow:

1. Initate Green Agent with task:
    Intake:
    1. Task deifniton
    2. Constraints
    3. Holdout data for evaluation with labels
2. Input docker container with model and defined entyrpoint where you input the validation data, and method for getting predictions, I guess it also eneeds a way to iddentify when done. 
3. While running on validation data, evaluate research
4. Combine for final scores
