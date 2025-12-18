# Predicting Marketing Campaign Success with Imbalanced Learning

> **"Can we accurately predict which customers will say *yes* to a marketing offer—even when almost everyone says *no*?"**  
> This project tackles that challenge using advanced ML pipelines, imbalance handling, and explainability.

---

## Results at a Glance

| Model                | F1-Score | ROC-AUC | PR-AUC | Recall (Class 1) |
|---------------------|----------|---------|--------|------------------|
| Logistic Regression | 0.51     | 0.89    | 0.478  | 0.79             |
| SMOTE + RandomForest| 0.55     | 0.91    | 0.538  | 0.55             |
| **SMOTE + XGBoost** | **0.56** | **0.918**| **0.568**| **0.54**          |

**Threshold Optimization (`≈ 0.33`)** boosted F1-score by improving the balance between precision and recall.

---

## Business Impact

> “Accurately identifying *who to contact* saves both time and marketing budget.”

Thanks to our final model (SMOTE + XGBoost + F1-optimized threshold), we can:

- Capture more **true responders**
- **Reduce wasted outreach**
- Make smarter decisions with **data-backed segmentation**

---

## Technical Highlights

- **Imbalance Handling:** SMOTE oversampling + `class_weight='balanced'`
- **Robust Evaluation:** Stratified 5-Fold CV + threshold optimization
- **Explainability:** SHAP to interpret model decisions
- **ML Pipelines:** Clean preprocessing + modeling with `Pipeline`
- **Metric Suite:** ROC-AUC, PR-AUC, F1-score, Recall
- **Bootstrap CI:** for statistical comparison of ROC-AUC
- **Feature Engineering:** categorical encoding, scaling, SMOTE integration
- **Data Source:** UCI Bank Marketing Dataset

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yourusername/bank-marketing-ml.git
cd bank-marketing-ml

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook marketing_prediction.ipynb
