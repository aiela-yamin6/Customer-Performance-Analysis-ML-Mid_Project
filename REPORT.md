# Customer Campaign and Spending Report

## Problem and data

This dataset uses the 2240 rows, 29 columns customer performance dataset. The cleaned analysis table has 2,236 rows and 28 columns after removing invalid ages and the single income sentinel, plus dropping identifiers, raw signup date, and constant contact/revenue columns. Income has 24 missing values; the target response is imbalanced (about 15% positive). We ask two questions: how much will a customer spend across six product categories, and how likely are they to accept a campaign?

## Preparation

Marital values `Absurd`, `YOLO`, and `Alone` are mapped to `Single`. Age is 2014 minus birth year; tenure is measured to 2014-12-31 with day-first date parsing. Total spend sums the six product spend fields, and total purchases sums the four purchase counts. Regression excludes all six spend components and the spend total from its predictors. The fixed 70/30 split uses seed 42; classification is stratified. Median imputation, encoding, and scaling are fit within train-only pipelines.

## Regression results

| Regressor | Test R² | Test RMSE |
|---|---:|---:|
| Simple Linear Regression (Income only) | 0.6926 | 334.08 |
| Linear Regression | 0.8299 | 248.53 |
| Ridge | 0.8299 | 248.50 |
| Lasso | 0.8311 | 247.66 |
| Polynomial Regression (degree 3) | 0.7563 | 297.45 |
| Decision Tree | 0.8205 | 255.29 |
| Random Forest | 0.8764 | 211.85 |

Random Forest is the best among the supported regressors on this holdout and is saved as `models/best_regressor.joblib`. The dashboard can recalculate all metrics, including RMSE, MAE, and train metrics.

## Classification results

| Classifier | Test ROC-AUC |
|---|---:|
| Logistic Regression | 0.8937 |
| Decision Tree | 0.7853 |
| Random Forest | 0.8903 |
| AdaBoost | 0.8882 |
| Gradient Boosting | 0.8997 |
| SVM | 0.8842 |
| Gaussian Naive Bayes | 0.8006 |

Gradient Boosting has the best test ROC-AUC (0.8997) among the compared classifiers and is saved as `models/best_classifier.joblib`. Campaign acceptance history may still be closely related to response, and random holdout ROC curves do not establish performance on future campaigns. For a future-campaign estimate, remove any predictors unavailable before the campaign and validate on a later time period. The dashboard also reports threshold-specific confusion matrices and imbalance comparisons.

## Business conclusion and limitations

The spend model offers a useful prioritization estimate, while response probability gives a ranking signal and a tunable decision threshold. Campaign teams should select that threshold using the cost of missed responders and the capacity for extra contact. This is a single historical dataset and one random holdout; no causal campaign effect is estimated. Model metrics and preprocessing effects are available live in the app. The supplied dataset lacks a separate customer identifier for future row-level validation and campaign-time metadata.

