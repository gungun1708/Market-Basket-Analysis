from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

from sales_analysis import clean_sales_data, load_sales_data


APP_DIR = Path(__file__).parent
DATASET_PATH = APP_DIR / "sales_data.csv"
CURRENCY_FORMATTER = FuncFormatter(lambda value, _: f"${value:,.0f}")
CHART_COLOR = "#2673b8"

st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {background-color: #f5f7fb;}
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e8f0;
        padding: 16px 18px; border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sales Analysis Dashboard")
st.markdown(
    "Explore revenue, order volume, customer segments, and product performance. "
    "Use the filters to compare the parts of the business that matter to you."
)

try:
    default_data = load_sales_data(DATASET_PATH)
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the sales dataset: {error}")
    st.stop()

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload a sales CSV (optional)",
        type=["csv"],
        help="Required columns: Order ID, Order Date, Product Name, Category, "
        "Region, Customer Segment, Quantity, and Unit Price.",
    )
    st.download_button(
        "Download sales dataset",
        data=default_data.dataframe.to_csv(index=False).encode("utf-8"),
        file_name="sales_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()
    st.header("Filters")

try:
    if uploaded_file is None:
        sales_data = default_data
        source_label = DATASET_PATH.name
    else:
        uploaded_data = pd.read_csv(uploaded_file)
        sales_data = clean_sales_data(uploaded_data)
        source_label = uploaded_file.name
except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not read the uploaded sales dataset: {error}")
    st.stop()

sales = sales_data.dataframe
date_min = sales["Order Date"].min().date()
date_max = sales["Order Date"].max().date()

with st.sidebar:
    selected_categories = st.multiselect(
        "Product category",
        options=sorted(sales["Category"].unique()),
        default=sorted(sales["Category"].unique()),
    )
    selected_regions = st.multiselect(
        "Region",
        options=sorted(sales["Region"].unique()),
        default=sorted(sales["Region"].unique()),
    )
    selected_segments = st.multiselect(
        "Customer segment",
        options=sorted(sales["Customer Segment"].unique()),
        default=sorted(sales["Customer Segment"].unique()),
    )
    selected_dates = st.date_input(
        "Order date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

if isinstance(selected_dates, (tuple, list)):
    start_date = selected_dates[0]
    end_date = selected_dates[-1] if len(selected_dates) > 1 else selected_dates[0]
else:
    start_date = end_date = selected_dates

filtered = sales.loc[
    sales["Category"].isin(selected_categories)
    & sales["Region"].isin(selected_regions)
    & sales["Customer Segment"].isin(selected_segments)
    & sales["Order Date"].dt.date.between(start_date, end_date)
].copy()

st.caption(
    f"Source: {source_label} · {len(sales):,} cleaned rows · "
    f"{sales_data.duplicate_rows:,} duplicate rows removed · "
    f"{sales_data.dropped_rows:,} incomplete or invalid rows removed"
)

if filtered.empty:
    st.warning("No sales match the selected filters. Adjust the sidebar filters.")
    st.stop()

total_revenue = float(filtered["Sales Revenue"].sum())
total_orders = int(filtered["Order ID"].nunique())
total_quantity = int(filtered["Quantity"].sum())
average_order_value = total_revenue / total_orders if total_orders else 0.0

kpi_columns = st.columns(4)
kpi_columns[0].metric("Total Revenue", f"${total_revenue:,.2f}")
kpi_columns[1].metric("Total Orders", f"{total_orders:,}")
kpi_columns[2].metric("Total Quantity Sold", f"{total_quantity:,}")
kpi_columns[3].metric("Average Order Value", f"${average_order_value:,.2f}")

monthly_revenue = (
    filtered.assign(Month=filtered["Order Date"].dt.to_period("M").dt.to_timestamp())
    .groupby("Month", as_index=False)["Sales Revenue"]
    .sum()
)
category_revenue = (
    filtered.groupby("Category", as_index=False)["Sales Revenue"]
    .sum()
    .sort_values("Sales Revenue", ascending=False)
)
region_revenue = (
    filtered.groupby("Region", as_index=False)["Sales Revenue"]
    .sum()
    .sort_values("Sales Revenue", ascending=False)
)
segment_revenue = filtered.groupby("Customer Segment")["Sales Revenue"].sum()
product_revenue = (
    filtered.groupby("Product Name")["Sales Revenue"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

st.subheader("Monthly Revenue Trend")
figure, axis = plt.subplots(figsize=(12, 4))
axis.plot(
    monthly_revenue["Month"],
    monthly_revenue["Sales Revenue"],
    color=CHART_COLOR,
    marker="o",
    linewidth=2.5,
)
axis.set_title("Revenue by Month")
axis.set_xlabel("Month")
axis.set_ylabel("Sales revenue (USD)")
axis.yaxis.set_major_formatter(CURRENCY_FORMATTER)
axis.grid(axis="y", alpha=0.25)
figure.autofmt_xdate()
figure.tight_layout()
st.pyplot(figure, use_container_width=True)
plt.close(figure)

left_chart, right_chart = st.columns(2)
with left_chart:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(category_revenue["Category"], category_revenue["Sales Revenue"], color=CHART_COLOR)
    axis.set_title("Category-wise Revenue")
    axis.set_xlabel("Category")
    axis.set_ylabel("Sales revenue (USD)")
    axis.yaxis.set_major_formatter(CURRENCY_FORMATTER)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with right_chart:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(region_revenue["Region"], region_revenue["Sales Revenue"], color="#42a58a")
    axis.set_title("Region-wise Sales")
    axis.set_xlabel("Region")
    axis.set_ylabel("Sales revenue (USD)")
    axis.yaxis.set_major_formatter(CURRENCY_FORMATTER)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

left_chart, right_chart = st.columns(2)
with left_chart:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.pie(
        segment_revenue,
        labels=segment_revenue.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.get_cmap("Set2").colors[: len(segment_revenue)],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    axis.set_title("Customer Segment Distribution (Revenue)")
    axis.axis("equal")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with right_chart:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(product_revenue.index, product_revenue.values, color="#e59345")
    axis.set_title("Top 10 Products by Revenue")
    axis.set_xlabel("Sales revenue (USD)")
    axis.set_ylabel("Product")
    axis.xaxis.set_major_formatter(CURRENCY_FORMATTER)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

best_product = filtered.groupby("Product Name")["Sales Revenue"].sum().idxmax()
best_category = category_revenue.iloc[0]["Category"]
best_region = region_revenue.iloc[0]["Region"]
best_month = monthly_revenue.loc[
    monthly_revenue["Sales Revenue"].idxmax(), "Month"
].strftime("%B %Y")

st.subheader("Performance Highlights")
insight_columns = st.columns(4)
insight_columns[0].info(f"**Top product:** {best_product}")
insight_columns[1].info(f"**Top category:** {best_category}")
insight_columns[2].info(f"**Top region:** {best_region}")
insight_columns[3].info(f"**Highest-revenue month:** {best_month}")

with st.expander("Summary statistics"):
    statistics = filtered[["Quantity", "Unit Price", "Sales Revenue"]].describe().T
    statistics["mean"] = np.mean(filtered[statistics.index], axis=0)
    st.dataframe(
        statistics,
        use_container_width=True,
        column_config={
            "mean": st.column_config.NumberColumn("mean", format="%.2f"),
            "std": st.column_config.NumberColumn("std", format="%.2f"),
            "min": st.column_config.NumberColumn("min", format="%.2f"),
            "max": st.column_config.NumberColumn("max", format="%.2f"),
        },
    )

st.subheader("Sales Dataset")
st.dataframe(
    filtered.sort_values("Order Date", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Order Date": st.column_config.DateColumn("Order Date", format="YYYY-MM-DD"),
        "Unit Price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
        "Sales Revenue": st.column_config.NumberColumn("Sales Revenue", format="$%.2f"),
    },
)
st.download_button(
    "Download filtered sales as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
