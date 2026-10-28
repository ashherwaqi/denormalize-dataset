"""Regional Sales Analytics & Forecasting app.

Pages:
  1. Data Analysis        - profiling, data quality, summaries, Actual vs Budget
  2. Data Visualization   - trends, products, regions, reps, distributions
  3. Predictive Analysis  - Sales / Units models, metrics, importance, what-if predictor
  4. Forecasting          - monthly forecasts with backtesting and confidence bands

Run:  streamlit run app.py
"""
import calendar
import glob
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (apply_filters, data_quality_report, load_sales,
                             schema_table, summary_by)
from src.forecaster import forecast, monthly_series
from src.ml_models import CATEGORICAL, NUMERICAL, TARGETS, predict_one, train_models

st.set_page_config(page_title="Regional Sales Analytics & Forecasting",
                   page_icon="📈", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
PAGES = ["1. Data Analysis", "2. Data Visualization",
         "3. Predictive Analysis", "4. Forecasting"]
METRIC_COL = {"Sales": "sales", "Units Sold": "units", "Profit": "profit"}


# ----------------------------------------------------------------- helpers
def human(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f} B"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f} M"
    if ax >= 1e3:
        return f"{x / 1e3:,.1f} K"
    return f"{x:,.0f}"


def find_default_file():
    candidates = (glob.glob(os.path.join(BASE, "data", "*.xlsx"))
                  + glob.glob(os.path.join(BASE, "data", "*.csv"))
                  + glob.glob(os.path.join(BASE, "*.xlsx")))
    return candidates[0] if candidates else None


@st.cache_data(show_spinner="Reading and cleaning the dataset…")
def get_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    return load_sales(io.BytesIO(file_bytes), file_name)


@st.cache_resource(show_spinner="Training models on the full dataset…")
def train_cached(_df: pd.DataFrame, target: str, data_key: str):
    # `_df` has a leading underscore so Streamlit does not hash the frame;
    # `data_key` invalidates the cache when a different file is loaded.
    return train_models(_df, target)


# ----------------------------------------------------------------- sidebar: data
with st.sidebar:
    st.title("📈 Sales Analytics")
    uploaded = st.file_uploader("Upload sales data (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded is not None:
    file_bytes, file_name = uploaded.getvalue(), uploaded.name
else:
    default_path = find_default_file()
    if default_path is None:
        st.error("No data found. Upload a file in the sidebar, or place the workbook "
                 "in the `data/` folder.")
        st.stop()
    with open(default_path, "rb") as fh:
        file_bytes = fh.read()
    file_name = os.path.basename(default_path)

data_key = f"{file_name}-{len(file_bytes)}"

try:
    df = get_data(file_bytes, file_name)
except ValueError as err:
    st.error(f"Could not read the file: {err}")
    st.stop()

# ----------------------------------------------------------------- sidebar: nav + filters
with st.sidebar:
    page = st.radio("Go to", PAGES)
    st.divider()
    st.subheader("Filters")
    st.caption("Applied to Data Analysis and Visualization.")
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    picked = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    start, end = (picked[0], picked[1]) if isinstance(picked, (tuple, list)) and len(picked) == 2 \
        else (dmin, dmax)
    scenarios = st.multiselect("Scenario", sorted(df["scenario"].unique()),
                               default=sorted(df["scenario"].unique()))
    categories = st.multiselect("Category", sorted(df["category"].unique()),
                                default=sorted(df["category"].unique()))
    cities = st.multiselect("City", sorted(df["city"].unique()),
                            default=sorted(df["city"].unique()))

filtered = apply_filters(df, start, end, scenarios, categories, cities)
st.caption(f"Data source: **{file_name}** · {len(df):,} rows loaded · "
           f"{len(filtered):,} rows after filters")

if filtered.empty:
    st.warning("No rows match the current filters. Widen the selection in the sidebar.")
    st.stop()

# ======================================================================
# PAGE 1 — DATA ANALYSIS
# ======================================================================
if page == PAGES[0]:
    st.header("1. Data Analysis")

    c = st.columns(5)
    c[0].metric("Transactions", f"{len(filtered):,}")
    c[1].metric("Total sales", human(filtered["sales"].sum()))
    c[2].metric("Total profit", human(filtered["profit"].sum()))
    c[3].metric("Profit margin", f"{filtered['profit'].sum() / filtered['sales'].sum() * 100:.1f}%")
    c[4].metric("Units sold", human(filtered["units"].sum()))

    tab_over, tab_qual, tab_stats, tab_group, tab_budget = st.tabs(
        ["Overview", "Data quality", "Summary statistics", "Group analysis", "Actual vs Budget"])

    with tab_over:
        st.subheader("Data preview")
        st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)
        st.subheader("Schema")
        st.dataframe(schema_table(filtered), use_container_width=True, hide_index=True)

    with tab_qual:
        st.subheader("Integrity and business-rule checks")
        st.dataframe(data_quality_report(filtered), use_container_width=True, hide_index=True)
        st.info("Sales equals Units × Sale price on every row, so Sales is an exact "
                "outcome of those two values. The predictive models therefore do not "
                "use Units or Sales as inputs, which would leak the answer.")

    with tab_stats:
        st.subheader("Descriptive statistics (numeric columns)")
        num_cols = ["units", "sales", "cost_of_sales", "sale_price", "cost_price",
                    "profit", "margin_pct", "age", "tenure_years"]
        num_cols = [c for c in num_cols if c in filtered.columns]
        st.dataframe(filtered[num_cols].describe().T.round(2), use_container_width=True)
        st.subheader("Correlation matrix")
        corr = filtered[num_cols].corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"),
                        use_container_width=True)

    with tab_group:
        dim = st.selectbox("Group by", ["category", "product", "brand", "city",
                                        "salesperson", "warranty", "designation"])
        table = summary_by(filtered, dim)
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "transactions": st.column_config.NumberColumn(format="%d"),
                "units": st.column_config.NumberColumn(format="%d"),
                "sales": st.column_config.NumberColumn(format="%.0f"),
                "cost_of_sales": st.column_config.NumberColumn(format="%.0f"),
                "profit": st.column_config.NumberColumn(format="%.0f"),
                "avg_sale_price": st.column_config.NumberColumn(format="%.0f"),
                "margin_pct": st.column_config.NumberColumn(format="%.1f%%"),
                "share_of_sales_pct": st.column_config.NumberColumn(format="%.1f%%"),
            })

    with tab_budget:
        st.subheader("Actual vs Budget by year")
        st.caption("Each year compares the Actual and Budget scenarios. A positive variance "
                   "means actual sales beat the budget.")
        yr = filtered.pivot_table(index="year", columns="scenario", values="sales",
                                  aggfunc="sum").reindex(columns=["Actual", "Budget"])
        yr = yr.dropna(how="all")
        if yr.isna().any().any():
            st.info("Select both scenarios in the sidebar to see variance.")
        else:
            yr["variance_pct"] = (yr["Actual"] / yr["Budget"] - 1) * 100
        st.dataframe(yr.reset_index(), use_container_width=True, hide_index=True,
                     column_config={"Actual": st.column_config.NumberColumn(format="%.0f"),
                                    "Budget": st.column_config.NumberColumn(format="%.0f"),
                                    "variance_pct": st.column_config.NumberColumn("Variance %",
                                                                                  format="%.1f%%")})

# ======================================================================
# PAGE 2 — DATA VISUALIZATION
# ======================================================================
elif page == PAGES[1]:
    st.header("2. Data Visualization")

    t_trend, t_prod, t_region, t_dist = st.tabs(
        ["Trends", "Products & categories", "Regions & sales reps", "Distributions"])

    with t_trend:
        monthly = (filtered.groupby(["year_month", "scenario"])["sales"].sum()
                   .reset_index())
        fig = px.line(monthly, x="year_month", y="sales", color="scenario",
                      title="Monthly sales: Actual vs Budget",
                      labels={"year_month": "Month", "sales": "Sales"})
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            yc = filtered.groupby(["year", "category"])["sales"].sum().reset_index()
            fig = px.bar(yc, x="year", y="sales", color="category", barmode="stack",
                         title="Yearly sales by category")
            fig.update_yaxes(tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pv = filtered.pivot_table(index="year", columns="month", values="sales",
                                      aggfunc="sum")
            pv.columns = [calendar.month_abbr[m] for m in pv.columns]
            fig = px.imshow(pv, aspect="auto", color_continuous_scale="Blues",
                            title="Seasonality: sales by year and month",
                            labels=dict(color="Sales"))
            st.plotly_chart(fig, use_container_width=True)

    with t_prod:
        top_n = st.slider("Number of products to show", 5, 30, 15)
        prod = summary_by(filtered, "product").head(top_n)
        fig = px.bar(prod.sort_values("sales"), x="sales", y="product", orientation="h",
                     color="category", title=f"Top {top_n} products by sales")
        fig.update_xaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            cat = summary_by(filtered, "category")
            fig = px.pie(cat, names="category", values="sales", hole=0.45,
                         title="Share of sales by category")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(summary_by(filtered, "product"), x="sales", y="profit",
                             size="units", color="category", hover_name="product",
                             title="Sales vs profit by product (bubble = units)")
            fig.update_xaxes(tickformat="~s")
            fig.update_yaxes(tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)

    with t_region:
        geo = (filtered.groupby("city")
               .agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                    sales=("sales", "sum"))
               .reset_index())
        geo["size_px"] = 20 + 80 * geo["sales"] / geo["sales"].max()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("City locations (bubble = sales)")
            st.map(geo, latitude="latitude", longitude="longitude", size="size_px")
        with c2:
            fig = px.bar(geo.sort_values("sales"), x="sales", y="city", orientation="h",
                         title="Sales by city")
            fig.update_xaxes(tickformat="~s")
            st.plotly_chart(fig, use_container_width=True)

        city_m = filtered.groupby(["year_month", "city"])["sales"].sum().reset_index()
        fig = px.line(city_m, x="year_month", y="sales", color="city",
                      title="Monthly sales by city")
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

        rep = summary_by(filtered, "salesperson")
        fig = px.bar(rep.sort_values("sales"), x="sales", y="salesperson", orientation="h",
                     color="share_of_sales_pct", color_continuous_scale="Teal",
                     title="Sales by salesperson")
        fig.update_xaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

    with t_dist:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(filtered, x="sales", nbins=60, log_x=True,
                               title="Distribution of transaction sales (log scale)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(filtered, x="category", y="margin_pct", color="category",
                         title="Profit margin % by category")
            st.plotly_chart(fig, use_container_width=True)
        fig = px.box(filtered, x="category", y="sale_price", color="category", log_y=True,
                     title="Sale price by category (log scale)")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# PAGE 3 — PREDICTIVE ANALYSIS
# ======================================================================
elif page == PAGES[2]:
    st.header("3. Predictive Analysis")
    st.caption("Models are trained on the full dataset (sidebar filters do not apply here). "
               "Results are measured on a held-out 20% test set.")

    target_label = st.radio("What do you want to predict?", list(TARGETS.keys()), horizontal=True)
    target = TARGETS[target_label]
    res = train_cached(df, target, data_key)
    metrics = res["metrics"]

    st.subheader("Model performance (test set)")
    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True,
                 column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "R²": st.column_config.NumberColumn(format="%.3f"),
                                "MAPE %": st.column_config.NumberColumn(format="%.1f%%")})
    best_r2 = metrics.loc[metrics["Model"] == res["main_name"], "R²"].iloc[0]
    if target == "units":
        st.warning(f"Units are hard to predict from these attributes (R² = {best_r2:.2f}). "
                   "Treat unit forecasts as rough guidance, not precise estimates.")
    else:
        st.success(f"The main model explains {best_r2:.0%} of the variation in transaction "
                   "sales on unseen data. Sale price is the dominant driver, which is expected "
                   "because Sales ≈ Units × Price.")

    c1, c2 = st.columns(2)
    with c1:
        samp = pd.DataFrame({"actual": res["y_test"].values, "predicted": res["pred_main"]})
        samp = samp.sample(min(3000, len(samp)), random_state=1)
        fig = px.scatter(samp, x="actual", y="predicted", opacity=0.35, log_x=True, log_y=True,
                         title=f"Actual vs predicted {target_label.lower()} (test sample)")
        lo, hi = samp[["actual", "predicted"]].min().min(), samp[["actual", "predicted"]].max().max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                                 line=dict(dash="dash", color="grey")))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        imp = res["importance"].head(12).sort_values("importance")
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="Feature importance (permutation, main model)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("What-if predictor")
    st.caption("Choose a scenario below to estimate the outcome of a single transaction.")

    model = res["models"][res["main_name"]]
    c1, c2, c3 = st.columns(3)
    product = c1.selectbox("Product", sorted(df["product"].unique()), key="w_prod")
    prow = df[df["product"] == product]
    sale_price = c1.number_input("Sale price", min_value=0.0,
                                 value=float(prow["sale_price"].median()),
                                 key=f"sp_{product}")
    cost_price = c1.number_input("Cost price", min_value=0.0,
                                 value=float(prow["cost_price"].median()),
                                 key=f"cp_{product}")

    city = c2.selectbox("City", sorted(df["city"].unique()), key="w_city")
    salesperson = c2.selectbox("Salesperson", sorted(df["salesperson"].unique()), key="w_rep")
    srow = df[df["salesperson"] == salesperson]
    sale_date = c2.date_input("Sale date", value=df["date"].max().date(), key="w_date")

    warranty = c3.selectbox("Warranty", sorted(df["warranty"].unique()), key="w_war")
    scenario = c3.selectbox("Scenario", ["Actual", "Budget"], key="w_scn")

    ts = pd.Timestamp(sale_date)
    inputs = {
        "product": product,
        "category": prow["category"].mode().iloc[0],
        "brand": prow["brand"].mode().iloc[0],
        "warranty": warranty,
        "city": city,
        "salesperson": salesperson,
        "designation": srow["designation"].mode().iloc[0],
        "scenario": scenario,
        "sale_price": sale_price,
        "cost_price": cost_price,
        "age": float(srow["age"].median()),
        "tenure_years": max(0.0, (ts - srow["doj"].min()).days / 365.25),
        "year": ts.year,
        "month": ts.month,
        "quarter": ts.quarter,
        "day_of_week": ts.dayofweek,
    }
    pred = predict_one(model, inputs, res["categorical"], res["numerical"])

    r1, r2, r3 = st.columns(3)
    if target == "sales":
        r1.metric("Predicted sales (one transaction)", f"{pred:,.0f}")
        r2.metric("Implied units at this price", f"{pred / sale_price:,.2f}" if sale_price else "–")
        r3.metric("Historical average for this product",
                  f"{prow['sales'].mean():,.0f}")
    else:
        r1.metric("Predicted units (one transaction)", f"{pred:,.2f}")
        r2.metric("Historical average units for this product", f"{prow['units'].mean():,.2f}")
        r3.metric("Typical units range (5th–95th pct)",
                  f"{prow['units'].quantile(0.05):.0f} – {prow['units'].quantile(0.95):.0f}")
    st.caption("Predictions for a single transaction are noisy. Use them to compare scenarios "
               "(for example, different prices or cities), not as exact figures.")

# ======================================================================
# PAGE 4 — FORECASTING
# ======================================================================
elif page == PAGES[3]:
    st.header("4. Forecasting")
    st.caption("Monthly forecasts use the full history for the chosen scenario. Sidebar filters "
               "do not apply here, so the forecast stays stable as you explore other pages.")

    c1, c2, c3 = st.columns(3)
    metric_label = c1.selectbox("Metric", list(METRIC_COL.keys()))
    scenario = c2.selectbox("Scenario", ["Actual", "Budget"],
                            help="Actual is used for forecasting. Budget is the plan.")
    level = c3.radio("Series", ["Total", "Category", "City"], horizontal=True)

    group_col, group_val = None, None
    if level == "Category":
        group_col, group_val = "category", st.selectbox("Category", sorted(df["category"].unique()))
    elif level == "City":
        group_col, group_val = "city", st.selectbox("City", sorted(df["city"].unique()))

    c4, c5 = st.columns(2)
    horizon = c4.slider("Forecast horizon (months)", 3, 24, 12)
    holdout = c5.slider("Backtest window (months held out for testing)", 6, 18, 12)

    metric_col = METRIC_COL[metric_label]
    series = monthly_series(df, metric_col, scenario, group_col, group_val)
    label = f"{metric_label} – {scenario}" + (f" – {group_val}" if group_val else " – Total")

    try:
        res = forecast(series, horizon=horizon, holdout=holdout)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    fc = res["forecast"]
    total_fc = fc["forecast"].sum()
    last_h = series.iloc[-horizon:].sum()
    growth = (total_fc / last_h - 1) * 100 if last_h else np.nan
    best_row = res["backtest"].iloc[0]

    m = st.columns(4)
    m[0].metric(f"Forecast total (next {horizon} mo)", human(total_fc))
    m[1].metric(f"Actual total (last {horizon} mo)", human(last_h))
    m[2].metric("Change vs last period", f"{growth:+.1f}%")
    m[3].metric("Backtest error (MAPE)", f"{best_row['MAPE %']:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual history",
                             line=dict(color="#1f77b4")))
    if scenario == "Actual" and metric_col == "sales":
        bud = monthly_series(df, metric_col, "Budget", group_col, group_val)
        fig.add_trace(go.Scatter(x=bud.index, y=bud.values, name="Budget (plan)",
                                 line=dict(color="grey", dash="dot")))
    fig.add_trace(go.Scatter(x=res["holdout_pred"].index, y=res["holdout_pred"].values,
                             name=f"Backtest fit ({res['best_model']})",
                             line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], fill="tonexty",
                             fillcolor="rgba(255,127,14,0.2)", line=dict(width=0),
                             name="Approx. 80% range"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast",
                             line=dict(color="#ff7f0e", width=3)))
    fig.update_layout(title=f"{label}: history and {horizon}-month forecast",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(tickformat="~s")
    st.plotly_chart(fig, use_container_width=True)

    if res["best_model"] == "Seasonal naive (same month last year)":
        st.caption("The seasonal naive model repeats last year's monthly pattern, so its "
                   "forecast total equals the last 12 months and the change is 0%.")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Model comparison (backtest)")
        st.dataframe(res["backtest"].round(2), use_container_width=True, hide_index=True,
                     column_config={"MAPE %": st.column_config.NumberColumn(format="%.2f%%"),
                                    "MAE": st.column_config.NumberColumn(format="%.0f"),
                                    "RMSE": st.column_config.NumberColumn(format="%.0f")})
        st.success(f"Selected model: **{res['best_model']}** (lowest backtest error). "
                   "It was refit on all history before forecasting.")
    with right:
        st.subheader("Forecast table")
        out = fc.copy()
        st.dataframe(out, use_container_width=True, hide_index=True,
                     column_config={
                         "date": st.column_config.DateColumn("Month", format="MMM YYYY"),
                         "forecast": st.column_config.NumberColumn(format="%.0f"),
                         "lower": st.column_config.NumberColumn("Lower (80%)", format="%.0f"),
                         "upper": st.column_config.NumberColumn("Upper (80%)", format="%.0f"),
                     })
        st.download_button("Download forecast (CSV)", fc.to_csv(index=False).encode(),
                           file_name=f"forecast_{metric_col}_{scenario}.csv", mime="text/csv")

    with st.expander("How the forecast works"):
        st.markdown(
            "- **Backtesting:** the last N months are hidden, each model forecasts them, and "
            "the model with the lowest MAPE is chosen. This is an honest check because the "
            "models never see the test months.\n"
            "- **Models:** seasonal naive repeats last year; trend + seasonality fits a linear "
            "trend with month effects; lag + ridge uses recent values and month effects.\n"
            "- **Range:** the approximate 80% band comes from the backtest error and widens "
            "with the forecast horizon. It is a guide, not a guarantee.\n"
            "- **Limits:** six years of monthly history supports seasonal patterns but not "
            "rare shocks such as policy changes or stock-outs. Use the forecast alongside "
            "business knowledge."
        )
