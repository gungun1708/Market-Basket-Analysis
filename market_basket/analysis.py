"""Transaction cleaning and Apriori association-rule mining."""

from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift"]


@dataclass
class TransactionData:
    records: pd.DataFrame
    basket_matrix: pd.DataFrame
    transaction_count: int
    unique_product_count: int
    item_entry_count: int
    dropped_missing_rows: int
    duplicate_rows: int


@dataclass
class AnalysisResults:
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame
    rules_found: int


def read_csv_dataset(file: BinaryIO) -> pd.DataFrame:
    """Read a CSV while preserving transaction IDs such as ``001``."""
    try:
        return pd.read_csv(file, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The uploaded file is not a valid CSV.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must use a supported text encoding, such as UTF-8.") from exc


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split()).casefold()


def prepare_transactions(
    dataframe: pd.DataFrame,
    transaction_column: str,
    item_column: str,
) -> TransactionData:
    """Clean long-format transaction rows and build a boolean basket matrix."""
    if dataframe.empty:
        raise ValueError("The dataset has no transaction rows.")
    if transaction_column == item_column:
        raise ValueError("Choose separate columns for transaction IDs and product names.")
    missing_columns = [
        column
        for column in (transaction_column, item_column)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Column(s) not found: {', '.join(missing_columns)}.")

    records = dataframe[[transaction_column, item_column]].copy()
    missing_mask = records[transaction_column].isna() | records[item_column].isna()
    for column in (transaction_column, item_column):
        records[column] = records[column].map(
            lambda value: "" if pd.isna(value) else " ".join(str(value).split())
        )
    blank_mask = records[transaction_column].eq("") | records[item_column].eq("")
    dropped_missing_rows = int((missing_mask | blank_mask).sum())
    records = records.loc[~(missing_mask | blank_mask)].copy()
    if records.empty:
        raise ValueError("No valid rows remain after removing blank transaction IDs and products.")

    records[transaction_column] = records[transaction_column].map(_normalize_text)
    records[item_column] = records[item_column].map(_normalize_text)
    duplicate_mask = records.duplicated([transaction_column, item_column], keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    records = records.loc[~duplicate_mask].reset_index(drop=True)

    basket_matrix = pd.crosstab(
        records[transaction_column], records[item_column]
    ).gt(0)
    return TransactionData(
        records=records,
        basket_matrix=basket_matrix,
        transaction_count=int(basket_matrix.shape[0]),
        unique_product_count=int(basket_matrix.shape[1]),
        item_entry_count=len(records),
        dropped_missing_rows=dropped_missing_rows,
        duplicate_rows=duplicate_rows,
    )


def mine_association_rules(
    transactions: TransactionData,
    min_support: float = 0.05,
    min_confidence: float = 0.2,
    min_lift: float = 1.0,
    max_rules: int = 100,
) -> AnalysisResults:
    """Find frequent itemsets and filter association rules by the chosen thresholds."""
    if not 0 < min_support <= 1:
        raise ValueError("Minimum support must be greater than 0 and at most 1.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum confidence must be between 0 and 1.")
    if min_lift < 0:
        raise ValueError("Minimum lift cannot be negative.")
    if max_rules < 1:
        raise ValueError("Maximum displayed rules must be at least 1.")

    itemsets = apriori(
        transactions.basket_matrix,
        min_support=min_support,
        use_colnames=True,
    )
    if itemsets.empty or not itemsets["itemsets"].map(len).ge(2).any():
        return AnalysisResults(
            frequent_itemsets=itemsets,
            rules=pd.DataFrame(columns=RULE_COLUMNS),
            rules_found=0,
        )

    rules = association_rules(
        itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )
    if rules.empty:
        return AnalysisResults(
            frequent_itemsets=itemsets,
            rules=pd.DataFrame(columns=RULE_COLUMNS),
            rules_found=0,
        )

    rules = rules.loc[rules["lift"] >= min_lift, RULE_COLUMNS].copy()
    rules["antecedents"] = rules["antecedents"].map(
        lambda products: ", ".join(sorted(products))
    )
    rules["consequents"] = rules["consequents"].map(
        lambda products: ", ".join(sorted(products))
    )
    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    rules_found = len(rules)
    return AnalysisResults(
        frequent_itemsets=itemsets,
        rules=rules.head(max_rules).reset_index(drop=True),
        rules_found=rules_found,
    )
