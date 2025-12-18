# Research Report: SpamHam

To optimize SMS spam classification, focus on balancing high precision with low-latency inference.

**1. Recommended Architectures**
*   **Baseline:** Multinomial Naive Bayes or Logistic Regression with TF-IDF for rapid iteration.
*   **SOTA:** **DistilBERT** or **TinyBERT**. These Transformers provide superior contextual understanding of short-form text while remaining computationally efficient for real-time production environments.

**2. Key Feature Engineering**
*   **Metadata:** Extract message length, punctuation density, and counts of URLs, phone numbers, or currency symbols.
*   **Textual:** Use character-level n-grams (3-5) to handle "leetspeak" and obfuscation (e.g., "v1agra").
*   **Imbalance:** Use **Cost-Sensitive Learning** (class weighting) rather than SMOTE to maintain the integrity of the natural data distribution.

**3. Common Pitfalls**
*   **Data Leakage:** Ensure identical messages or unique sender IDs do not bridge the train-test split.
*   **Over-preprocessing:** Avoid aggressive lemmatization; capitalization and "SMS-speak" (slang/shorthand) are often strong spam indicators.

**Cross-Domain Transfer Learning**
Leverage **Transfer Learning** from models pre-trained on massive, diverse corpora (e.g., RoBERTa). Although trained on formal text, the underlying linguistic hierarchies transfer exceptionally well; fine-tuning allows the model to adapt general semantic knowledge to the specific nuances of informal mobile messaging.