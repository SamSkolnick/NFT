# Research Report: Titanic

To optimize survival prediction, prioritize **Gradient Boosted Decision Trees (GBDT)**—specifically **CatBoost** (for native categorical handling) or **LightGBM**. For state-of-the-art results, implement a **Stacked Ensemble** using a Logistic Regression meta-learner over GBDT, Random Forest, and a simple MLP.

**Key Feature Engineering:**
*   **Title Extraction:** Parse `Name` for social status (e.g., Master, Noble).
*   **Deck Mapping:** Extract the first letter from `Cabin` to infer vertical location.
*   **Family Cohorts:** Create a `Family_Survival` proxy by grouping `Surname` and `Ticket`. If a family member died, the probability for others drops significantly.
*   **Binning:** Use non-linear binning for `Age` and `Fare` to capture non-monotonic relationships.

**Common Pitfalls:**
*   **Data Leakage:** Information in `Ticket` or `Fare` can inadvertently reveal survival outcomes through shared group identifiers.
*   **Overfitting:** With $N \approx 891$, complex models overfit noise. Use **Stratified 10-Fold CV** and avoid deep trees.

**Cross-Domain Transfer:**
Incorporate methodologies from **Epidemiology (Clinical Survival Analysis)** and **Credit Risk Modeling**. Treat the "survival" label as a "hazard rate" influenced by environmental exposure (deck location), mirroring how researchers model patient outcomes in oncology or default risk in finance.