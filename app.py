"""Streamlit dashboard for interactive market basket analysis."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from market_basket.analysis import (
    mine_association_rules,
    prepare_transactions,
    read_csv_dataset,
)


APP_DIR = Path(__file__).parent
DEFAULT_DATASET = APP_DIR / "data" / "sample_transactions.csv"

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e8f0;
        padding: 16px 18px; border-radius: 12px;
    }
    .project-footer {
        border-top: 1px solid #dfe5ee; color: #64748b;
        margin-top: 3rem; padding: 1rem 0; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _format_items(value: object) -> str:
    return ", ".join(sorted(str(item) for item in value))


st.title("Market Basket Analysis")
st.markdown(
    "Discover products that frequently occur together and translate association "
    "rules into practical merchandising decisions."
)
st.caption(
    "An Apriori-based analysis of transaction-level data for the Chandigarh "
    "University project portfolio."
)

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload a transaction CSV",
        type=["csv"],
        help="The file must contain one transaction ID column and one product/item column.",
    )
    st.download_button(
        "Download sample dataset",
        data=DEFAULT_DATASET.read_bytes(),
        file_name=DEFAULT_DATASET.name,
        mime="text/csv",
        use_container_width=True,
    )

try:
    if uploaded_file is None:
        with DEFAULT_DATASET.open("rb") as dataset_file:
            raw_data = read_csv_dataset(dataset_file)
        source_label = DEFAULT_DATASET.relative_to(APP_DIR).as_posix()
    else:
        raw_data = read_csv_dataset(uploaded_file)
        source_label = uploaded_file.name
except (OSError, ValueError) as error:
    st.error(f"Could not load the transaction dataset: {error}")
    st.stop()

if len(raw_data.columns) < 2:
    st.error("The dataset must contain at least two columns.")
    st.stop()

with st.sidebar:
    st.header("Columns")
    transaction_column = st.selectbox(
        "Transaction ID column",
        options=list(raw_data.columns),
        index=0,
        help="Identifies the basket or order containing the item.",
    )
    item_options = [column for column in raw_data.columns if column != transaction_column]
    item_column = st.selectbox(
        "Product/item column",
        options=item_options,
        index=0,
        help="Contains the product or item name.",
    )
    st.divider()
    st.header("Apriori filters")
    min_support = st.slider(
        "Minimum support",
        min_value=0.01,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help="Share of transactions containing an itemset.",
    )
    min_confidence = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Likelihood of the consequent when the antecedent occurs.",
    )
    min_lift = st.number_input(
        "Minimum lift",
        min_value=0.0,
        value=1.0,
        step=0.1,
        help="Values above 1 indicate a positive association.",
    )
    max_rules = st.slider("Rules to display", 5, 100, 25, 5)

try:
    transactions = prepare_transactions(raw_data, transaction_column, item_column)
    analysis = mine_association_rules(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_rules=max_rules,
    )
except ValueError as error:
    st.error(f"Unable to analyze this dataset: {error}")
    st.info("Upload long-format data with one row per item in a transaction.")
    st.stop()

st.caption(
    f"Source: {source_label} · {transactions.dropped_missing_rows:,} invalid rows "
    f"removed · {transactions.duplicate_rows:,} duplicate item entries removed"
)

kpi_columns = st.columns(4)
kpi_columns[0].metric("Transactions analyzed", f"{transactions.transaction_count:,}")
kpi_columns[1].metric("Unique products", f"{transactions.unique_product_count:,}")
kpi_columns[2].metric(
    "Frequent itemsets",
    f"{len(analysis.frequent_itemsets):,}",
    help="Item combinations meeting the selected minimum support.",
)
kpi_columns[3].metric(
    "Association rules",
    f"{analysis.rules_found:,}",
    help="Rules meeting the selected confidence and lift thresholds.",
)

if analysis.frequent_itemsets.empty:
    st.warning(
        "No frequent itemsets meet the selected support threshold. "
        "Lower minimum support to explore more combinations."
    )

frequency = (
    transactions.basket_matrix.sum()
    .sort_values(ascending=False)
    .rename("Transactions")
    .reset_index()
    .rename(columns={"col_0": "Product", "index": "Product"})
)
if "Product" not in frequency:
    frequency = frequency.rename(columns={frequency.columns[0]: "Product"})
frequency["Share"] = frequency["Transactions"] / transactions.transaction_count

left, right = st.columns((1.1, 1))
with left:
    st.subheader("Most frequently purchased products")
    frequency_chart = px.bar(
        frequency.head(15).sort_values("Transactions"),
        x="Transactions",
        y="Product",
        orientation="h",
        text="Transactions",
        color="Transactions",
        color_continuous_scale=["#b7d7f0", "#1769aa"],
    )
    frequency_chart.update_layout(
        height=470,
        coloraxis_showscale=False,
        xaxis_title="Transactions containing product",
        yaxis_title=None,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(frequency_chart, use_container_width=True)
with right:
    st.subheader("What the metrics mean")
    st.info(
        "**Support** measures how often an itemset appears in all baskets. "
        "**Confidence** measures how often the consequent appears when the "
        "antecedent is present. **Lift** compares that result with random "
        "co-occurrence: lift above 1 indicates a positive association."
    )
    st.dataframe(
        frequency.assign(
            Share=frequency["Share"].map(lambda value: f"{value:.1%}")
        )[["Product", "Transactions", "Share"]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Strongest association rules")
if analysis.rules.empty:
    st.warning(
        "No association rules meet all selected thresholds. "
        "Try lowering confidence or lift."
    )
else:
    rules_display = analysis.rules.copy()
    rules_display["Rule"] = (
        rules_display["antecedents"] + " → " + rules_display["consequents"]
    )
    rules_display["Support"] = rules_display["support"].map(lambda value: f"{value:.1%}")
    rules_display["Confidence"] = rules_display["confidence"].map(
        lambda value: f"{value:.1%}"
    )
    rules_display["Lift"] = rules_display["lift"].map(lambda value: f"{value:.2f}")
    st.dataframe(
        rules_display[["Rule", "Support", "Confidence", "Lift"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download association rules",
        data=analysis.rules.to_csv(index=False).encode("utf-8"),
        file_name="association_rules.csv",
        mime="text/csv",
    )

    rule_chart_data = analysis.rules.head(15).copy()
    rule_chart_data["Rule"] = (
        rule_chart_data["antecedents"] + " → " + rule_chart_data["consequents"]
    )
    rule_chart = px.scatter(
        rule_chart_data,
        x="support",
        y="confidence",
        size="lift",
        color="lift",
        hover_name="Rule",
        color_continuous_scale="Teal",
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
    )
    rule_chart.update_layout(
        height=430,
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(rule_chart, use_container_width=True)

with st.expander("Frequent itemsets"):
    if analysis.frequent_itemsets.empty:
        st.info("No frequent itemsets to display.")
    else:
        itemsets_display = analysis.frequent_itemsets.copy()
        itemsets_display["Items"] = itemsets_display["itemsets"].map(_format_items)
        itemsets_display["Support"] = itemsets_display["support"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            itemsets_display[["Items", "Support"]].sort_values(
                "Support", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Business interpretation"):
    if analysis.rules.empty:
        st.write(
            "There are no qualifying rules at the current thresholds. This is a "
            "valid analytical result: increase the amount of transaction history "
            "or relax the thresholds before making merchandising decisions."
        )
    else:
        strongest = analysis.rules.iloc[0]
        st.write(
            f"The strongest displayed rule is **{strongest['antecedents']} → "
            f"{strongest['consequents']}** with {strongest['confidence']:.1%} "
            f"confidence and {strongest['lift']:.2f} lift. Possible applications "
            "include bundle offers, shelf placement, cross-selling prompts, and "
            "recommendation-system features. These associations indicate "
            "co-occurrence, not causation, and should be validated with business "
            "experiments."
        )

st.download_button(
    "Download cleaned transaction records",
    data=transactions.records.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_transactions.csv",
    mime="text/csv",
)

st.markdown(
    '<div class="project-footer">Built by Gungun Gupta · '
    '<a href="https://github.com/gungun1708/Market-Basket-Analysis" '
    'target="_blank">GitHub repository</a></div>',
    unsafe_allow_html=True,
)
