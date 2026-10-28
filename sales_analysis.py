"""Sales dataset generation, cleaning, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Product Name",
    "Category",
    "Region",
    "Customer Segment",
    "Quantity",
    "Unit Price",
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["Sales Revenue"]

PRODUCTS = [
    ("Wireless Mouse", "Electronics", 24.99),
    ("Bluetooth Speaker", "Electronics", 59.99),
    ("USB-C Hub", "Electronics", 39.99),
    ("Desk Lamp", "Home & Office", 34.50),
    ("Ergonomic Chair", "Home & Office", 189.00),
    ("Notebook Set", "Home & Office", 12.99),
    ("Coffee Maker", "Appliances", 79.00),
    ("Air Fryer", "Appliances", 119.00),
    ("Blender", "Appliances", 64.50),
    ("Running Shoes", "Sports", 89.99),
    ("Yoga Mat", "Sports", 28.00),
    ("Water Bottle", "Sports", 18.50),
    ("Skin Care Set", "Beauty", 42.00),
    ("Hair Dryer", "Beauty", 54.99),
    ("Scented Candle", "Home & Office", 16.00),
    ("Fitness Tracker", "Electronics", 99.00),
]
REGIONS = ["North", "South", "East", "West", "Central"]
CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Small Business", "Enterprise"]


@dataclass
class CleanSalesData:
    dataframe: pd.DataFrame
    dropped_rows: int = 0
    duplicate_rows: int = 0


def create_sample_sales_data(
    order_count: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """Create reproducible multi-year sales data suitable for dashboard demos."""
    if order_count < 1:
        raise ValueError("Order count must be at least one.")

    random = np.random.default_rng(seed)
    product_indices = random.integers(0, len(PRODUCTS), size=order_count)
    product_names = np.array([product[0] for product in PRODUCTS])
    categories = np.array([product[1] for product in PRODUCTS])
    base_prices = np.array([product[2] for product in PRODUCTS])
    quantities = random.integers(1, 7, size=order_count)
    unit_prices = np.round(
        base_prices[product_indices] * random.uniform(0.9, 1.1, size=order_count),
        2,
    )
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        random.integers(0, 730, size=order_count),
        unit="D",
    )

    data = pd.DataFrame(
        {
            "Order ID": [f"ORD-{number:05d}" for number in range(1, order_count + 1)],
            "Order Date": dates,
            "Product Name": product_names[product_indices],
            "Category": categories[product_indices],
            "Region": random.choice(REGIONS, size=order_count),
            "Customer Segment": random.choice(CUSTOMER_SEGMENTS, size=order_count),
            "Quantity": quantities,
            "Unit Price": unit_prices,
        }
    )
    data["Sales Revenue"] = data["Quantity"] * data["Unit Price"]
    return data[OUTPUT_COLUMNS].sort_values("Order Date").reset_index(drop=True)


def clean_sales_data(dataframe: pd.DataFrame) -> CleanSalesData:
    """Validate sales columns, remove invalid/duplicate records, and recalculate revenue."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Missing required sales column(s): " + ", ".join(missing_columns) + "."
        )

    cleaned = dataframe[REQUIRED_COLUMNS].copy()
    original_count = len(cleaned)
    duplicate_mask = cleaned.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    for column in ("Order ID", "Product Name", "Category", "Region", "Customer Segment"):
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    cleaned["Order Date"] = pd.to_datetime(cleaned["Order Date"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Unit Price"] = pd.to_numeric(cleaned["Unit Price"], errors="coerce")
    valid_rows = cleaned.dropna(subset=REQUIRED_COLUMNS).copy()
    valid_rows = valid_rows.loc[
        (valid_rows["Quantity"] > 0) & (valid_rows["Unit Price"] >= 0)
    ].copy()
    dropped_rows = original_count - duplicate_rows - len(valid_rows)

    if valid_rows.empty:
        raise ValueError("No valid sales rows remain after cleaning the dataset.")

    valid_rows["Quantity"] = valid_rows["Quantity"].astype(int)
    valid_rows["Sales Revenue"] = valid_rows["Quantity"] * valid_rows["Unit Price"]
    valid_rows["Order Date"] = valid_rows["Order Date"].dt.normalize()
    return CleanSalesData(
        dataframe=valid_rows[OUTPUT_COLUMNS].reset_index(drop=True),
        dropped_rows=dropped_rows,
        duplicate_rows=duplicate_rows,
    )


def load_sales_data(path: str | Path) -> CleanSalesData:
    """Load and clean the requested CSV, creating a sample file if it is absent."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        create_sample_sales_data().to_csv(dataset_path, index=False)
    raw_data = pd.read_csv(dataset_path)
    return clean_sales_data(raw_data)
