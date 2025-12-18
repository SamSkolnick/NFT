# HeartDisease Evaluation Report

## Research
**Research Summary: Heart Disease Classification**

**1. Model Architectures**
For tabular clinical data, **Gradient Boosted Decision Trees (XGBoost, CatBoost)** remain state-of-the-art due to their handling of non-linearities and missing values. For large-scale EHR (Electronic Health Records), utilize **TabNet** or **Transformer-based tabular models** to capture complex feature interactions.

**2. Feature Engineering & Cross-Domain Analogies**
*   **Domain Ratios:** Calculate medical indices like the Total Cholesterol/HDL ratio and Pulse Pressure.
*   **NLP Analogy:** Treat clinical history as a "sentence" using **Med2Vec (Word2Vec variant)**; embed ICD codes to capture latent semantic relationships between comorbidities.
*   **Fraud Detection Analogy:** Apply **Anomaly Detection (Isolation Forests)**. Heart disease often presents as a physiological "deviation" from a healthy baseline, similar to detecting fraudulent transactions.
*   **Scaling:** Use **Quantile Transformation** to handle skewed clinical distributions (e.g., Triglycerides).

**3. Common Pitfalls**
*   **Target Leakage:** Ensure predictors like "Nitroglycerin prescription" aren't included, as they imply a prior diagnosis.
*   **Selection Bias:** Models trained on hospital data may fail on general populations (Spectrum Bias).
*   **Interpretability:** Clinical adoption requires **SHAP or LIME**; a "black box" prediction is useless without identifying the risk drivers (e.g., hypertension vs. genetics).

## Performance
```json
{
  "0": {
    "precision": 0.8181818181818182,
    "recall": 0.75,
    "f1-score": 0.782608695652174,
    "support": 12.0
  },
  "1": {
    "precision": 0.7272727272727273,
    "recall": 0.8,
    "f1-score": 0.7619047619047619,
    "support": 10.0
  },
  "accuracy": 0.7727272727272727,
  "macro avg": {
    "precision": 0.7727272727272727,
    "recall": 0.775,
    "f1-score": 0.7722567287784678,
    "support": 22.0
  },
  "weighted avg": {
    "precision": 0.7768595041322315,
    "recall": 0.7727272727272727,
    "f1-score": 0.7731978166760776,
    "support": 22.0
  }
}
```

## Files
- **Model Path:** /Users/samuelskolnick/MLEngineer/SolverAgent/model/model.pkl
- **Code Path:** /Users/samuelskolnick/MLEngineer/SolverAgent/models/model_98e6c31fa3f24570badf9b5a2b0d76cd.py
