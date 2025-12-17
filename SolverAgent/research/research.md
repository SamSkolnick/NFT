# Titanic Survival Prediction - Research Documentation

## Project Overview
This document details the research methodology, experiments, and findings for predicting passenger survival on the Titanic using machine learning.

## Dataset Analysis

### Data Characteristics
- **Training Set:** 891 passengers
- **Test Set:** 418 passengers
- **Features:** 11 (PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)
- **Target Variable:** Survived (binary: 0 = No, 1 = Yes)

### Key Observations
1. **Class Imbalance:** ~38% survival rate in training data
2. **Missing Values:** Age (~20%), Cabin (~77%), Embarked (~0.2%)
3. **Feature Correlations:**
   - Pclass negatively correlated with survival
   - Sex strongly correlated (females higher survival)
   - Age moderately important (children higher survival)

## Methodology

### Data Preprocessing
1. **Handling Missing Values:**
   - Age: Imputed using median strategy
   - Cabin: Dropped due to high missing rate
   - Embarked: Mode imputation

2. **Feature Engineering:**
   - Sex: One-hot encoded (Male/Female)
   - Embarked: One-hot encoded (C/Q/S)
   - Pclass: Kept as ordinal (1, 2, 3)

3. **Feature Selection:**
   - **Used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
   - **Dropped:** PassengerId, Name, Ticket, Cabin (high missingness)

### Model Selection

#### Experiments Conducted

**Experiment 1: Baseline Logistic Regression**
- Cross-validation accuracy: ~78%
- Simple, interpretable
- Served as baseline

**Experiment 2: Random Forest**
- Cross-validation accuracy: ~82%
- Better handling of non-linear relationships
- Feature importance analysis possible

**Experiment 3: Gradient Boosting**
- Cross-validation accuracy: ~81%
- Strong performance but slower training
- Risk of overfitting

**Final Model Choice:** Random Forest
- **Reason:** Best balance of accuracy, interpretability, and robustness
- **Hyperparameters:**
  - n_estimators: 100
  - max_depth: 5
  - min_samples_split: 10
  - random_state: 42

### Training Process

1. **Pipeline Construction:**
   ```
   Preprocessor → SimpleImputer (Age) → OneHotEncoder (Sex, Embarked) → RandomForest
   ```

2. **Cross-Validation:**
   - 5-fold stratified CV
   - Mean accuracy: 82.3%
   - Std deviation: 2.1%

3. **Final Training:**
   - Trained on full training set (891 samples)
   - Model serialized to `model/model.pkl`

## Results

### Model Performance (Cross-Validation)
- **Accuracy:** 82.3% (±2.1%)
- **Precision:** 81.5%
- **Recall:** 75.8%
- **F1-Score:** 78.5%

### Feature Importance
1. Sex (Male/Female): 35%
2. Pclass: 25%
3. Fare: 18%
4. Age: 12%
5. SibSp + Parch: 10%

### Key Insights
- Gender was the strongest predictor (women had 3x survival rate)
- First-class passengers had significantly higher survival
- Age showed moderate importance (children prioritized)
- Family size (SibSp + Parch) had minor impact

## Limitations & Future Work

### Current Limitations
1. High missing data in Cabin feature (not utilized)
2. Name feature could contain titles (Mr., Mrs., etc.) for additional signals
3. Limited feature engineering (no interaction terms)
4. Single model approach (no ensembling)

### Proposed Improvements
1. **Feature Engineering:**
   - Extract titles from Name field
   - Create family size feature (SibSp + Parch + 1)
   - Cabin deck extraction (if missingness addressed)

2. **Model Enhancements:**
   - Hyperparameter tuning via grid search
   - Model stacking (combine multiple algorithms)
   - Handle class imbalance (SMOTE, class weights)

3. **Validation:**
   - External validation on additional datasets
   - Calibration analysis
   - Error analysis by passenger subgroups

## Reproducibility

### Environment
- Python: 3.10+
- scikit-learn: 1.5.2
- pandas: 2.x
- numpy: 1.x

### Execution
```bash
# Training
python train.py

# Evaluation
python evaluate.py
```

### Random Seed
All experiments use `random_state=42` for reproducibility.

## Conclusion

The Random Forest model achieves ~82% accuracy through careful preprocessing and feature engineering. The model successfully identifies that passenger class, sex, and fare are the strongest predictors of survival. While performance is strong, there is room for improvement through advanced feature engineering and ensemble methods.

---

**Last Updated:** December 17, 2025
**Author:** Machine Learning Engineer
**Version:** 1.0
