# Research Report: ExamScores

For predicting student scores, prioritize **Gradient Boosted Decision Trees (XGBoost or CatBoost)** as they excel with heterogeneous tabular data and handle missing values robustly. If the dataset includes longitudinal records (sequence of quiz scores), implement a **Temporal Fusion Transformer (TFT)** to capture long-term dependencies.

**Feature Engineering:**
*   **Learning Velocity:** Calculate the first derivative of scores over time to measure improvement rate.
*   **Engagement Variance:** Rolling standard deviation of study hours to identify inconsistent learners.
*   **Target Encoding:** Apply to high-cardinality categoricals like School/Teacher IDs.
*   **Interaction Terms:** Multiply "Prior Performance" by "Attendance Rate" to capture non-linear synergies.

**Common Pitfalls:**
*   **Data Leakage:** Ensure features from the target exam itself (or post-exam metrics) are strictly excluded.
*   **Selection Bias:** Account for "survivorship bias" where lower-performing students drop out before the final exam.
*   **Overfitting:** Use nested cross-validation, as educational datasets are often small and prone to noise.

**Cross-Domain Strategy:**
Frame this as a **Predictive Maintenance** or **Customer Lifetime Value (CLV)** problem. Academically "at-risk" students mirror "failing components." Utilizing pre-trained tabular models (like **TabPFN**) provides strong priors from diverse industrial datasets, significantly improving accuracy on small student cohorts.