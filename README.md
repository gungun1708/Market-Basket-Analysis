# Sales Analysis Dashboard

A Streamlit dashboard for exploring sales revenue, orders, products, regions,
customer segments, and monthly trends. It uses Pandas and NumPy for data
preparation and analysis, and Matplotlib for its charts. The first run creates a
reproducible sample dataset in `sales_data.csv` if the file is not already
present.

## Requirements and installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the dashboard

From the project directory:

```bash
streamlit run app.py
```

The dashboard opens in your browser. Use the sidebar to filter by category,
region, customer segment, and order date. You can optionally upload another
sales CSV or download the sample and filtered data.

## Sales CSV format

The dashboard expects these columns:

```text
Order ID, Order Date, Product Name, Category, Region, Customer Segment,
Quantity, Unit Price
```

`Sales Revenue` is calculated as `Quantity × Unit Price`; supplied values in
that column are recalculated. Exact duplicate rows and records with missing or
invalid required values are removed. A reproducible 1,200-order sample covering
2024 and 2025 is generated if `sales_data.csv` does not exist.

## Tests

Run all project tests with:

```bash
python -m pytest
```

The original market-basket analysis helpers and their tests remain in
`market_basket/` and `tests/test_analysis.py`.
