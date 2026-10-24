# Telco Customer Churn Prediction

Identifying at-risk telecom customers with Logistic Regression, Decision Tree and Random Forest.

## Overview

Customer churn (customers leaving the service) directly reduces revenue, and keeping an existing customer is typically cheaper than acquiring a new one. This project analyzes the IBM Telco Customer Churn dataset to:

1. Understand which customer characteristics are associated with churn.
2. Build and compare classification models, focusing on correctly identifying the churn class.
3. Handle class imbalance and tune the decision threshold for a retention use case.
4. Turn the findings into business recommendations.

## Dataset

- **Source:** IBM sample dataset, [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns, no missing values and no duplicated rows
- **Target:** `Churn` (about 26.5% of customers churned, so the classes are imbalanced)
- **Features:** customer profile (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account information (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`) and subscribed services (phone, internet and add-on services)

## Workflow

1. **Data cleaning:** converted `TotalCharges` to numeric (11 blank values belong to customers with tenure 0), merged `No phone service` and `No internet service` into `No`, dropped `customerID`, encoded the target.
2. **EDA:** target distribution, feature distributions, churn rate by numerical and categorical features, correlation analysis (Pearson, Spearman, Cramér's V).
3. **Modeling:** Dummy baseline, Logistic Regression, class-weighted Logistic Regression, Decision Tree and Random Forest, built as scikit-learn pipelines (preprocessing is fitted on the training data only).
4. **Evaluation:** stratified 5-fold cross-validation, validation-set comparison, ROC curves, overfitting check, threshold tuning on the validation set, and a single final evaluation on the test set.
5. **Interpretation:** Logistic Regression coefficients.

The data is split 60% train, 20% validation and 20% test (stratified). The test set is used only once, at the end.

## Key Findings

- **Tenure:** churn is highest among new customers, about 53% in the first 6 months versus about 10% for customers with 49-72 months.
- **Contract:** month-to-month customers churn at about 43%, compared with about 11% (one-year) and about 3% (two-year).
- **Internet service and payment:** fiber optic customers (about 42%) and electronic check payers (about 45%) churn the most.
- **Add-on services:** customers without online security or tech support churn at about 31%, versus about 15% for those with them.
- **No clear effect:** `gender`, `PhoneService` and `MultipleLines`.

These are associations, not proof of causes.

## Model Results

Churn-class metrics on the validation set at the default 0.5 threshold:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (always predicts "stayed") | 0.00 | 0.00 | 0.00 | 0.50 |
| Logistic Regression | 0.70 | 0.59 | 0.64 | 0.86 |
| Random Forest | 0.65 | 0.48 | 0.56 | 0.83 |
| Decision Tree | 0.50 | 0.51 | 0.51 | 0.67 |

- Accuracy is misleading here: the baseline reaches about 73.5% accuracy but finds no churners, so the models are judged on churn precision, recall, F1 and ROC-AUC.
- Logistic Regression performs best, while the Decision Tree and Random Forest overfit the training data.
- At the default threshold, Logistic Regression finds only about half of the churners on the test set (recall about 0.53). Class weighting and threshold tuning address this by trading some precision for higher recall. The results of the selected model and threshold on the test set are in section 6.6 of the notebook.

## Recommendations

- Rank customers by predicted churn probability and prioritize retention efforts for the highest-risk ones.
- Focus onboarding and early check-ins on customers in their first year.
- Offer incentives that move month-to-month customers to longer contracts.
- Investigate why electronic check payers and fiber optic customers churn more, and whether bundling security or tech support helps.
- Test retention offers with A/B experiments before scaling them.

## Limitations and Next Steps

- Results come from one dataset and one train/validation/test split.
- The data has no time dimension (no information on when or why customers left).
- Hyperparameters were not tuned and probabilities were not calibrated.
- Next steps: hyperparameter tuning, gradient boosting, probability calibration, and choosing the threshold from a cost-benefit analysis (cost of a retention offer versus revenue lost per churned customer).

## Repository Structure

```
.
├── telcocustomerchurn_revised.ipynb         # full analysis
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # dataset (download from Kaggle)
└── README.md
```

## How to Run

```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn jupyter
jupyter notebook telcocustomerchurn_revised.ipynb
```

Place the CSV file in the same folder as the notebook, then run all cells (`Kernel` > `Restart & Run All`).

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn

## Author

Rini Ramadhan Naim - [LinkedIn](https://www.linkedin.com/in/rini-ramadhan-naim-1b0286429)
