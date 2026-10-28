# Market Basket Analysis Dashboard

An interactive Python application that applies the Apriori algorithm to
transaction-level data and turns frequent product combinations into
interpretable association rules. The project is designed as a portfolio and
Chandigarh University submission, with data cleaning, validation, visualization,
testing, and a Streamlit dashboard in one repository.

## Problem statement

Retailers need to understand which products are purchased together so they can
design bundles, improve product placement, and create more relevant
cross-selling recommendations. This project discovers those relationships from
historical transaction records without claiming that correlation proves
causation.

## Objectives

- Validate and clean long-format transaction records.
- Convert item-level rows into a transaction-by-product basket matrix.
- Discover frequent itemsets with Apriori.
- Generate and rank rules using support, confidence, and lift.
- Provide an interactive dashboard for threshold tuning and business
  interpretation.
- Keep the analysis reproducible, testable, and suitable for a new CSV upload.

## Dataset

The repository includes [`data/sample_transactions.csv`](data/sample_transactions.csv),
a small, transparent example dataset with `transaction_id` and `item` columns.
It contains 60 item rows across 20 transaction IDs and 8 products. It is
intended for demonstrating the workflow, not for representing real customers.
The dashboard does not silently generate customer or transaction data.

You can upload any legitimate long-format CSV with one row per item, such as:

```csv
transaction_id,item
T001,bread
T001,butter
T002,bread
```

The dashboard lets you select the transaction and product columns, so common
column names such as `Order ID`/`Product Name` are supported.

## Technologies

- Python 3.9+
- Pandas and NumPy for data preparation
- MLxtend for Apriori and association rules
- Plotly and Streamlit for interactive analytics
- Pytest for automated tests

## How the analysis works

1. Blank IDs/items and invalid rows are removed.
2. Duplicate transaction-item pairs are removed.
3. A boolean basket matrix is built, with transactions as rows and products as
   columns.
4. Apriori finds itemsets that meet the minimum support threshold.
5. Association rules are filtered by minimum confidence and lift.
6. Results are ranked by lift, confidence, and support.

### Metric definitions

- **Support:** the proportion of all transactions containing an itemset.
- **Confidence:** the proportion of transactions containing the antecedent that
  also contain the consequent.
- **Lift:** confidence divided by the consequent's overall support. Lift above
  1 suggests a positive association compared with random co-occurrence.

## Installation

Use Python 3.10 or newer for the smoothest local experience. Python 3.9 is
also supported by the source code.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

From the project directory:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit. Use the sidebar to upload a CSV,
select its columns, and tune minimum support, confidence, and lift.

## Run tests

```bash
python -m pytest
```

The tests cover transaction cleaning, normalization, duplicate handling,
validation, Apriori metric calculations, empty-result behavior, and the
preserved sales-analysis helpers.

## Screenshots

Add screenshots of the running dashboard here for a submission or portfolio
README, for example:

```text
docs/dashboard-overview.png
docs/association-rules.png
```

## Example business applications

- Bundle products with strong lift and practical support.
- Place frequently co-purchased products near each other.
- Trigger cross-sell recommendations at checkout.
- Prioritize experiments for promotions and product bundles.

## Limitations and future improvements

- The included dataset is intentionally small and illustrative; larger
  transaction history is needed for production decisions.
- Apriori can become expensive as the product catalogue grows; FP-Growth or
  FP-Max could improve scalability.
- Rules show association, not causation, and should be validated with controlled
  experiments.
- Future work could add category hierarchies, time-window analysis, user
  authentication, and exportable reports.

## Repository

GitHub: https://github.com/gungun1708/Market-Basket-Analysis
