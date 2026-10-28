# Regional Sales Analytics & Forecasting (Streamlit)

An interactive app built on the regional sales workbook (100,000 transactions,
Jan 2015 – Dec 2020, 30 products, 5 cities, 10 sales reps, Actual vs Budget).

| Page | What it does |
|---|---|
| 1. Data Analysis | KPIs, data preview, schema, integrity checks, descriptive stats, correlations, group summaries, Actual vs Budget variance |
| 2. Data Visualization | Monthly trends, yearly category mix, seasonality heatmap, top products, sales vs profit, city map, rep performance, distributions |
| 3. Predictive Analysis | Gradient-boosting model for transaction **Sales** or **Units Sold**, compared with a ridge baseline; feature importance; what-if predictor |
| 4. Forecasting | Monthly forecasts for Total / Category / City, with backtesting, model selection, 80% range and CSV export |

## Setup

```bash
pip install -r requirements.txt
mkdir -p data && cp "Denormalize_regional_sales_data.xlsx" data/
streamlit run app.py
```

You can also upload the workbook (or a CSV with the same columns) from the sidebar.

## Project layout

```
app.py                  Streamlit UI (4 pages, global filters)
src/data_loader.py      Header detection, cleaning, derived features, quality checks
src/ml_models.py        Sales / Units models, baseline, permutation importance, prediction
src/forecaster.py       Monthly series builder, 3 forecasters, backtest, forecast + bands
requirements.txt
```

## Modelling notes

- **Leakage control.** `Sales = Units × Sale price` exactly, so Units and Sales are never
  model inputs. Inputs are product, prices, city, rep, warranty, scenario and date features.
- **Main model:** `HistGradientBoostingRegressor` on a log-transformed target. Baseline:
  ridge regression on the same features.
- **Forecast selection:** seasonal naive, trend + month effects, and recursive lag + ridge.
  The model with the lowest MAPE on the last N held-out months is refit on all history.
- **Scenarios:** forecasting uses *Actual*. *Budget* is shown as a reference line only,
  because mixing the two would double-count transactions.

## Findings from the data

- Sales are driven almost entirely by sale price. Units sold are close to unpredictable
  from the available attributes (R² ≈ 0.06), so unit forecasts are rough guidance.
- Actual sales grew every year, from about 1.25 B in 2015 to about 1.83 B in 2020. Combined Actual + Budget grew from about 2.46 B to about 3.56 B over the same period.
- Laptops make up roughly two-thirds of sales.

## Assumptions

- The currency is not stated in the workbook, so values are shown without a currency symbol.
- Each row is one transaction. Cost of sales matches units × cost price to within 0.5%
  on all rows (rounding).
"# denormalize-dataset" 
