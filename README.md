# Retail Customer Purchase Prediction

This project fulfills the retail problem statement: it uses transaction history to estimate whether a customer will make another recorded purchase and provides personalized product recommendations.

## What is included

- `app.py` — a simple, interactive web app for the project demonstration.
- `train_model.py` — trains and saves the predictive model without opening the app.
- `src/retail_predictor.py` — data cleaning, feature engineering, model training, prediction, and recommendations.
- `Retail_Transactions_Dataset.csv` — the supplied dataset.
- `PROJECT_REPORT.md` — ready-to-use project methodology and limitations.

## Easiest way to run it (Windows)

1. Keep all files in this folder.
2. Double-click `run_project.bat`.
3. The first run installs the required Python packages. This may take a few minutes.
4. A browser page opens. Click **Train / update model** in the left panel.
5. Open **Customer prediction**, choose a customer, and view their purchase likelihood and product recommendations.

The model file is created at `models/purchase_predictor.joblib` after training. You do not need to create it yourself.

## Alternative: run from a terminal

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py --data Retail_Transactions_Dataset.csv --horizon 365
streamlit run app.py
```

If PowerShell blocks the activation script, use the provided `run_project.bat` instead.

## Prediction definition

The dataset has no existing “future purchase” column. The project therefore creates a transparent target:

> `Will_Purchase_In_Horizon = 1` when the same customer has another recorded transaction within the chosen number of days after a transaction; otherwise it is `0`.

The default horizon is 365 days. Rows too close to the end of the dataset are excluded from model training, because their future purchase outcome is unknown rather than negative.

## Recommendation method

Recommendations are a hybrid of:

1. Products the selected customer has purchased before (repeat-purchase signal).
2. Products commonly bought together with the customer's latest basket (cross-sell signal).
3. Products popular in a similar city, store type, and customer category (context signal).
4. Overall product popularity (fallback for customers with little history).

Each recommendation shows a plain-language explanation.

## Dataset limitation

The supplied data contains 1,000 valid transactions but only a small number of customers with repeat purchases. This is not enough to claim a production-quality return-purchase model. The application detects sparse positive examples and displays a low-confidence warning. For a real deployment, collect substantially more transactions per identifiable customer and use the predictions as decision support, with proper fairness, privacy, and business validation.
