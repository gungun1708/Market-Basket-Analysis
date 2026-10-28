import pandas as pd
import pytest

from sales_analysis import (
    OUTPUT_COLUMNS,
    clean_sales_data,
    create_sample_sales_data,
    load_sales_data,
)


def test_generated_dataset_has_expected_columns_and_revenue():
    dataset = create_sample_sales_data(order_count=25, seed=7)

    assert dataset.columns.tolist() == OUTPUT_COLUMNS
    assert len(dataset) == 25
    assert (
        dataset["Sales Revenue"]
        == dataset["Quantity"] * dataset["Unit Price"]
    ).all()
    assert dataset["Order Date"].dt.year.isin([2024, 2025]).all()


def test_cleaning_recalculates_revenue_and_removes_invalid_and_duplicate_rows():
    records = pd.DataFrame(
        [
            ["A1", "2025-01-03", " Lamp ", "Home", "West", "Consumer", 2, 10.0, 999],
            ["A1", "2025-01-03", " Lamp ", "Home", "West", "Consumer", 2, 10.0, 999],
            ["A2", None, "Book", "Office", "East", "Corporate", 1, 5.0, 5],
            ["A3", "bad-date", "Book", "Office", "East", "Corporate", 1, 5.0, 5],
        ],
        columns=OUTPUT_COLUMNS,
    )

    cleaned = clean_sales_data(records)

    assert len(cleaned.dataframe) == 1
    assert cleaned.dataframe.loc[0, "Product Name"] == "Lamp"
    assert cleaned.dataframe.loc[0, "Sales Revenue"] == 20.0
    assert cleaned.duplicate_rows == 1
    assert cleaned.dropped_rows == 2


def test_cleaning_rejects_missing_columns_or_no_valid_records():
    with pytest.raises(ValueError, match="Missing required sales column"):
        clean_sales_data(pd.DataFrame({"Order ID": ["A1"]}))

    invalid = pd.DataFrame(
        [["A1", "not-a-date", "Lamp", "Home", "West", "Consumer", 1, 10.0]],
        columns=OUTPUT_COLUMNS[:-1],
    )
    with pytest.raises(ValueError, match="No valid sales rows remain"):
        clean_sales_data(invalid)


def test_loader_creates_a_csv_when_missing(tmp_path):
    dataset_path = tmp_path / "sales_data.csv"

    loaded = load_sales_data(dataset_path)

    assert dataset_path.is_file()
    assert not loaded.dataframe.empty
    assert loaded.dataframe.columns.tolist() == OUTPUT_COLUMNS
