# Retail Customer Purchase Prediction — Project Notes

## Problem statement

Retailers find it difficult to predict customer buying behavior. The objective is to develop an AI system that analyzes transaction history, predicts future purchases, and produces personalized product recommendations.

## Solution overview

The solution is a Python and Streamlit application with two connected outputs:

1. **Purchase likelihood:** a class-balanced logistic-regression model estimates whether a customer will make another recorded purchase within a chosen future period.
2. **Product recommendation:** a hybrid recommendation engine ranks likely repeat purchases, basket complements, contextually popular items, and overall popular items.

## Data preparation

- Read the supplied CSV and ignore blank rows.
- Validate the required fields and remove records with missing transaction ID, customer name, or date.
- Convert dates from day-month-year format.
- Convert spend and item count to numeric values.
- Convert the list-like product basket to individual product names.
- Convert `Discount_Applied` to a numeric indicator.

## Feature engineering

For each customer transaction, the model uses:

- current spend, item count, discount use, city, store type, payment method, promotion, season, customer category, and primary product;
- number of transactions so far;
- average spend and average items so far;
- historical discount rate;
- days since the preceding purchase; and
- number of distinct products previously purchased.

All history features are calculated using information available on or before the transaction date, which avoids using a future transaction as an input feature.

## Target variable

The dataset does not contain a future-purchase label. The project creates `Will_Purchase_In_Horizon`:

- **1:** the same customer has another recorded transaction within the selected horizon (default: 365 days);
- **0:** no such transaction is observed, only when the full future horizon is visible in the dataset.

Transactions in the final horizon period are excluded from training because their outcome is not fully observable.

## Evaluation

When both target classes have enough examples, the application uses a stratified 75/25 train-test split and reports accuracy, precision, recall, F1 score, average precision, and ROC-AUC when applicable. The app reports a low-confidence warning when positive repeat-purchase examples are sparse.

## Important limitation and future work

Most customers in the supplied dataset have only one recorded transaction. This makes personalized return-purchase prediction statistically weak. The correct next step is to collect more repeat customer history, customer IDs that are stable over time, product-level quantities, and online behavior. With richer data, future work could compare gradient boosting models, tune the decision threshold based on campaign cost, and evaluate recommendations with click or conversion experiments.
