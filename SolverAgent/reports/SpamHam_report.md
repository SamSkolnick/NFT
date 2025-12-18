# SpamHam Evaluation Report

## Research
**Architecture:** For SOTA performance, use **DistilBERT** or **RoBERTa-small**; they capture semantic intent better than frequency-based models. If latency is critical, a **Linear SVM** or **XGBoost** trained on TF-IDF features serves as a robust baseline.

**Feature Engineering:** Implement **Character-level N-grams** (n=3-5) to counter "leetspeak" obfuscation (e.g., `S.P.A.M`). Integrate metadata: message length, punctuation density (e.g., `!!!`), and URL counts. Address class imbalance—typical in SMS—using **Cost-Sensitive Learning** or **Focal Loss** rather than oversampling, which can lead to overfitting on short sequences.

**Common Pitfalls:** Avoid heavy stop-word removal; in SMS, pronouns and specific verbs often signal urgency. Watch for **Data Leakage** via sender IDs or timestamps. Ensure the model handles Unicode/Emojis, as these are high-signal features in modern spam.

**Cross-Domain Strategy:** Treat SMS spam like **Genomic Sequence Analysis**: apply **K-mer counting** logic to identify "motifs" of malicious intent within obfuscated strings. Borrow from **Financial Fraud Detection**: treat classification as an adversarial anomaly detection problem. Use **Transfer Learning** from large-scale email datasets (Enron) to pre-train on the universal semantics of "phishing" before fine-tuning on SMS-specific brevity.

## Performance
```json
{
  "0": {
    "precision": 0.9871794871794872,
    "recall": 0.9948320413436692,
    "f1-score": 0.990990990990991,
    "support": 387.0
  },
  "1": {
    "precision": 0.9642857142857143,
    "recall": 0.9152542372881356,
    "f1-score": 0.9391304347826087,
    "support": 59.0
  },
  "accuracy": 0.984304932735426,
  "macro avg": {
    "precision": 0.9757326007326008,
    "recall": 0.9550431393159025,
    "f1-score": 0.9650607128867998,
    "support": 446.0
  },
  "weighted avg": {
    "precision": 0.9841509387473514,
    "recall": 0.984304932735426,
    "f1-score": 0.9841305138244112,
    "support": 446.0
  }
}
```

## Files
- **Model Path:** /Users/samuelskolnick/MLEngineer/SolverAgent/model/model.pkl
- **Code Path:** /Users/samuelskolnick/MLEngineer/SolverAgent/models/model_6d212e2f75c64c989abeac59e064cd8a.py
