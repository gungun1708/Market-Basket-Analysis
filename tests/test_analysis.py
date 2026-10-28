import io

import pandas as pd
import pytest

from market_basket.analysis import (
    mine_association_rules,
    prepare_transactions,
    read_csv_dataset,
)


def test_preprocessing_normalizes_and_deduplicates_items():
    dataframe = pd.DataFrame(
        {
            "transaction_id": [" 001 ", "001", "001", "002", None, "003"],
            "item": [" Bread  ", "bread", "MILK", " Butter ", "tea", "  "],
        }
    )

    prepared = prepare_transactions(dataframe, "transaction_id", "item")

    assert prepared.transaction_count == 2
    assert prepared.unique_product_count == 3
    assert prepared.item_entry_count == 3
    assert prepared.dropped_missing_rows == 2
    assert prepared.duplicate_rows == 1
    assert prepared.basket_matrix.loc["001", "bread"]
    assert prepared.basket_matrix.loc["001", "milk"]
    assert prepared.basket_matrix.loc["002", "butter"]


def test_apriori_rule_metrics_are_calculated_correctly():
    dataframe = pd.DataFrame(
        {
            "transaction_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "item": ["bread", "butter", "bread", "butter", "bread", "milk", "milk", "eggs"],
        }
    )
    prepared = prepare_transactions(dataframe, "transaction_id", "item")

    result = mine_association_rules(
        prepared,
        min_support=0.25,
        min_confidence=0.5,
        min_lift=0,
    )

    bread_to_butter = result.rules.query(
        "antecedents == 'bread' and consequents == 'butter'"
    ).iloc[0]
    assert bread_to_butter["support"] == pytest.approx(0.5)
    assert bread_to_butter["confidence"] == pytest.approx(2 / 3)
    assert bread_to_butter["lift"] == pytest.approx(4 / 3)


def test_empty_or_invalid_datasets_are_rejected():
    with pytest.raises(ValueError, match="no transaction rows"):
        prepare_transactions(pd.DataFrame(columns=["transaction_id", "item"]), "transaction_id", "item")

    with pytest.raises(ValueError, match="No valid rows remain"):
        prepare_transactions(
            pd.DataFrame({"transaction_id": [None], "item": ["bread"]}),
            "transaction_id",
            "item",
        )

    with pytest.raises(ValueError, match="Column"):
        prepare_transactions(pd.DataFrame({"id": ["1"]}), "id", "item")


def test_empty_csv_is_rejected():
    with pytest.raises(ValueError, match="CSV is empty"):
        read_csv_dataset(io.BytesIO(b""))


def test_thresholds_with_no_itemsets_or_rules_return_empty_results():
    single_item_transactions = prepare_transactions(
        pd.DataFrame({"transaction_id": ["1", "2"], "item": ["bread", "milk"]}),
        "transaction_id",
        "item",
    )
    no_itemsets = mine_association_rules(single_item_transactions, min_support=0.75)
    assert no_itemsets.frequent_itemsets.empty
    assert no_itemsets.rules.empty
    assert no_itemsets.rules_found == 0

    paired_transactions = prepare_transactions(
        pd.DataFrame(
            {
                "transaction_id": ["1", "1", "2", "3", "4"],
                "item": ["bread", "butter", "bread", "butter", "other"],
            }
        ),
        "transaction_id",
        "item",
    )
    no_rules = mine_association_rules(
        paired_transactions,
        min_support=0.25,
        min_confidence=0.5,
        min_lift=1.1,
    )
    assert not no_rules.frequent_itemsets.empty
    assert no_rules.rules.empty
    assert no_rules.rules_found == 0


def test_metrics_thresholds_are_validated():
    prepared = prepare_transactions(
        pd.DataFrame({"transaction_id": ["1"], "item": ["bread"]}),
        "transaction_id",
        "item",
    )
    with pytest.raises(ValueError, match="Minimum support"):
        mine_association_rules(prepared, min_support=0)
