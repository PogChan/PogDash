import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import calendar
from dateutil.relativedelta import relativedelta
import logging
import sys
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('spy_calendar.log')
    ]
)
logger = logging.getLogger('spy_calendar')

dash.register_page(__name__, path="/calendar", name="Calendar")

# -----------------------------
# Data Functions
# -----------------------------

def engineer_features(data):
    """Create comprehensive feature set for prediction"""
    data = data.copy()

    # Target variable
    data["target"] = (data["Daily_Return"] > 0).astype(int)

    # Momentum features
    for window in [5, 10, 20, 30]:
        data[f"Rolling_{window}_Return"] = data["Close"].pct_change(window) * 100

    # Volatility features
    for window in [5, 10]:
        data[f"Volatility_{window}"] = data["Daily_Return"].rolling(window).std()

    # Temporal features
    data["DayOfWeek"] = data.index.dayofweek
    data["Month"] = data.index.month

    # Lag features
    for lag in [1, 2]:
        data[f"Prev_{lag}_Return"] = data["Daily_Return"].shift(lag)

    # Volume features
    if "Volume" in data.columns:
        data["Vol_Change"] = data["Volume"].pct_change() * 100
        data["Vol_Ratio_5"] = data["Volume"] / data["Volume"].rolling(5).mean()

    # Technical indicators
    data["RSI_14"] = ta.rsi(data["Close"], length=14)
    fast = 5
    slow = 10
    signal = 3
    macd = ta.macd(data["Close"], fast=fast, slow=slow, signal=signal)
    macd_column = f"MACD_{fast}_{slow}_{signal}"
    macds_column = f"MACDs_{fast}_{slow}_{signal}"
    data["MACD"] = macd[macd_column]
    data["MACD_Signal"] = macd[macds_column]

    logger.info(data.head())
    return data

def train_predictive_model(data):
    """Train Random Forest model with time-series validation"""
    try:
        data = engineer_features(data)


        feature_columns = [
            "Rolling_5_Return", "Rolling_10_Return", "Rolling_20_Return", "Rolling_30_Return",
            "Volatility_5", "Volatility_10", "DayOfWeek", "Month",
            "Prev_1_Return", "Prev_2_Return", "RSI_14", "MACD", "MACD_Signal"
        ]

        if "Volume" in data.columns:
            feature_columns.extend(["Vol_Change", "Vol_Ratio_5"])

        train_data = data[feature_columns + ["target"]].dropna()
        feature_columns = [col for col in feature_columns if col in train_data.columns]

        X = train_data[feature_columns]
        y = train_data["target"]

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            cv_scores.append(model.score(X_test, y_test))

        logger.info(f"CV Scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        model.fit(X, y)

        # Feature importance analysis
        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        logger.info(f"Feature Importance:\n{importance}")

        return model, feature_columns

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return None, []

def add_predictions(data, model, feature_columns):
    """Add prediction probabilities and confidence metrics"""
    data = engineer_features(data)
    valid_rows = data[feature_columns].notna().all(axis=1)

    logger.info(f"Valid rows for prediction: {valid_rows.sum()} out of {len(data)}")
    if valid_rows.any():
        X = data.loc[valid_rows, feature_columns]
        data.loc[valid_rows, "pred_prob"] = model.predict_proba(X)[:, 1]
        data.loc[valid_rows, "pred_direction"] = model.predict(X)
        data.loc[valid_rows, "prediction_confidence"] = abs(data.loc[valid_rows, "pred_prob"] - 0.5) * 2

    return data


def calculate_seasonality_with_predictions(data):
    data = data.copy()
    data["Month"] = data.index.month
    data["Trading_Day"] = data.groupby([data.index.year, data.index.month]).cumcount() + 1

    # Log raw pred_prob stats before aggregation
    if "pred_prob" in data.columns:
        pred_probs = data["pred_prob"].dropna()
        if not pred_probs.empty:
            logger.info(f"Raw pred_prob stats: min={pred_probs.min():.2f}, max={pred_probs.max():.2f}, mean={pred_probs.mean():.2f}")
        else:
            logger.warning("No non-NaN pred_prob values before aggregation")

    agg_dict = {
        "Daily_Return": ["median", "count"],
        "pred_prob": "median"  # Changed to median
    }
    if "prediction_confidence" in data.columns:
        agg_dict["prediction_confidence"] = "median"  # Changed to median

    stats = data.groupby(["Month", "Trading_Day"]).agg(agg_dict)
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = [f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in stats.columns]

    stats = stats.reset_index()
    if "Daily_Return_median" in stats.columns:
        stats = stats.rename(columns={"Daily_Return_median": "median", "Daily_Return_count": "count"})

    # Log aggregated pred_prob
    if "pred_prob_median" in stats.columns:
        pred_probs_agg = stats["pred_prob_median"].dropna()
        logger.info(f"Aggregated pred_prob stats: min={pred_probs_agg.min():.2f}, max={pred_probs_agg.max():.2f}, mean={pred_probs_agg.mean():.2f}")


    # Calculate positive day statistics
    pos_counts = data[data["Daily_Return"] > 0].groupby(["Month", "Trading_Day"]).size().reset_index(name="pos_count")
    stats = pd.merge(stats, pos_counts, on=["Month", "Trading_Day"], how="left")
    stats["pos_count"] = stats["pos_count"].fillna(0)
    stats["pos_pct"] = (stats["pos_count"] / stats["count"] * 100).round(1)

    # Calculate negative day statistics
    neg_counts = data[data["Daily_Return"] < 0].groupby(["Month", "Trading_Day"]).size().reset_index(name="neg_count")
    stats = pd.merge(stats, neg_counts, on=["Month", "Trading_Day"], how="left")
    stats["neg_count"] = stats["neg_count"].fillna(0)
    stats["neg_pct"] = (stats["neg_count"] / stats["count"] * 100).round(1)

    # Calculate accuracy metrics if pred_direction is available
    if "pred_direction" in data.columns:
        # Check if directional prediction was correct (actual matches predicted)
        data["correct_prediction"] = ((data["Daily_Return"] > 0) == (data["pred_direction"] == 1))

        # Aggregate accuracy per trading day
        prediction_accuracy = data.groupby(["Month", "Trading_Day"])["correct_prediction"].mean().reset_index(name="prediction_accuracy")
        stats = pd.merge(stats, prediction_accuracy, on=["Month", "Trading_Day"], how="left")

        # Separate accuracy for up and down days
        up_days = data[data["Daily_Return"] > 0]
        if not up_days.empty:
            up_accuracy = up_days.groupby(["Month", "Trading_Day"])["correct_prediction"].mean().reset_index(name="up_prediction_accuracy")
            stats = pd.merge(stats, up_accuracy, on=["Month", "Trading_Day"], how="left")

        down_days = data[data["Daily_Return"] < 0]
        if not down_days.empty:
            down_accuracy = down_days.groupby(["Month", "Trading_Day"])["correct_prediction"].mean().reset_index(name="down_prediction_accuracy")
            stats = pd.merge(stats, down_accuracy, on=["Month", "Trading_Day"], how="left")

    logger.info(f"Calculated seasonality stats with enhanced predictions for {len(stats)} groups")
    return stats



def get_spy_data(start_date="2000-01-01", end_date=None):
    try:
        logger.info(f"Fetching SPY data from {start_date} to {end_date}")
        data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in data.columns]
            data = data.rename(columns={
                "Close_SPY": "Close",
                "Volume_SPY": "Volume",
                "High_SPY": "High",
                "Low_SPY": "Low",
                "Open_SPY": "Open"
            })

        data["Daily_Return"] = data["Close"].pct_change() * 100
        logger.info(f"Retrieved {len(data)} days of SPY data")
        return data

    except Exception as e:
        logger.error(f"Failed to retrieve SPY data: {str(e)}")
        return pd.DataFrame()


def filter_correction_periods(data, threshold=-15):
    """
    Improved version that provides more flexibility in filtering.
    This version can remove entire months where corrections occurred.
    """
    # Calculate rolling 30-day return (in %)
    data["Rolling_30_Return"] = data["Close"].pct_change(30) * 100

    # Identify correction periods
    is_correction = data["Rolling_30_Return"] < threshold

    # Find dates where corrections occurred
    correction_dates = data.index[is_correction]

    if len(correction_dates) == 0:
        return data  # No corrections found

    # Create a copy of data to avoid modifying the original
    filtered_data = data.copy()

    # Identify unique year-month combinations with corrections
    correction_months = set()
    for date in correction_dates:
        correction_months.add((date.year, date.month))

    # Create a mask for keeping data (exclude correction months)
    keep_mask = pd.Series(True, index=data.index)
    for year, month in correction_months:
        month_mask = (data.index.year == year) & (data.index.month == month)
        keep_mask = keep_mask & (~month_mask)

    # Return filtered data
    filtered_data = data[keep_mask]

    logger.info(f"Removed {len(data) - len(filtered_data)} days from {len(correction_months)} correction month(s)")

    return filtered_data


def calculate_seasonality(data):
    data = data.copy()
    data["Month"] = data.index.month
    # For each year-month, assign a trading day number (starting at 1)
    data["Trading_Day"] = data.groupby([data.index.year, data.index.month]).cumcount() + 1

    # Aggregate using median instead of mean
    stats = data.groupby(["Month", "Trading_Day"])["Daily_Return"].agg(["median", "count"]).reset_index()

    # Calculate positive counts and merge them
    pos_counts = data[data["Daily_Return"] > 0].groupby(["Month", "Trading_Day"]).size().reset_index(name="pos_count")
    stats = pd.merge(stats, pos_counts, on=["Month", "Trading_Day"], how="left")
    stats["pos_count"] = stats["pos_count"].fillna(0)
    stats["pos_pct"] = (stats["pos_count"] / stats["count"] * 100).round(1)

    # Calculate negative counts and percentage
    neg_counts = data[data["Daily_Return"] < 0].groupby(["Month", "Trading_Day"]).size().reset_index(name="neg_count")
    stats = pd.merge(stats, neg_counts, on=["Month", "Trading_Day"], how="left")
    stats["neg_count"] = stats["neg_count"].fillna(0)
    stats["neg_pct"] = (stats["neg_count"] / stats["count"] * 100).round(1)

    logger.info(f"Calculated seasonality stats with {len(stats)} unique month/trading day combinations")
    return stats

# Modified function to support additional data in the calendar
def generate_calendar_data(seasonality, month):
    logger.info(f"Generating enhanced calendar data for month {month}")
    month_data = seasonality[seasonality["Month"] == month]
    if month_data.empty:
        logger.warning(f"No data available for month {month}")
        max_trading_day = 0
    else:
        max_trading_day = int(month_data["Trading_Day"].max())
        logger.info(f"Month {month} has {max_trading_day} trading days")

    calendar_list = []
    for td in range(1, max_trading_day + 1):
        row = month_data[month_data["Trading_Day"] == td]
        if not row.empty:
            day_data = {
                "trading_day": td,
                "sample_size": int(row["count"].values[0])
            }

            # Add all available columns, safely handling possible missing ones
            for col in ["median", "pos_pct", "neg_pct", "pred_prob", "prediction_confidence",
                         "prediction_accuracy", "up_prediction_accuracy", "down_prediction_accuracy"]:
                if col in row.columns and not row[col].isna().all():
                    day_data[col] = row[col].values[0]
        else:
            day_data = {
                "trading_day": td,
                "median": np.nan,
                "pos_pct": np.nan,
                "neg_pct": np.nan,
                "sample_size": 0
            }
        calendar_list.append(day_data)
    return calendar_list

def create_calendar_layout(year, month, seasonality_df, min_sample=10, columns=5):
    logger.info(f"Creating calendar layout for {calendar.month_name[month]} {year}")
    cal_data = generate_calendar_data(seasonality_df, month)

    # Log pred_prob stats
    pred_probs = [day['pred_prob'] for day in cal_data if 'pred_prob' in day and day['pred_prob'] is not None]
    if pred_probs:
        logger.info(f"pred_prob stats in cal_data: min={min(pred_probs):.2f}, max={max(pred_probs):.2f}, mean={np.mean(pred_probs):.2f}")
    else:
        logger.info("No pred_prob values available in cal_data")

    header = html.Tr([html.Th(f"TD {i+1}", className="text-center custom-table-header") for i in range(columns)])
    rows = [header]
    cells = []
    for day_info in cal_data:
        if day_info["sample_size"] < min_sample or np.isnan(day_info.get("median", np.nan)):
            bg = "#1a1a1a"  # Dark gray for insufficient data
            text_color = "text-muted"
        else:
            median = day_info.get("median", 0)
            pos_pct = day_info.get("pos_pct", 0)
            neg_pct = day_info.get("neg_pct", 0)
            pred_prob = day_info.get("pred_prob", None)

            if pred_prob is not None and not np.isnan(pred_prob):
                # Color based on predicted probability
                if pred_prob >= 0.55:
                    if pred_prob >= 0.75:
                        bg = "#00ff00"  # Bright green
                    elif pred_prob >= 0.65:
                        bg = "#36ff00"  # Medium green
                    else:
                        bg = "#70d800"  # Light green
                    text_color = "text-success"
                elif pred_prob <= 0.45:
                    if pred_prob <= 0.25:
                        bg = "#ff0000"  # Bright red
                    elif pred_prob <= 0.35:
                        bg = "#ff3333"  # Medium red
                    else:
                        bg = "#ff6666"  # Light red
                    text_color = "text-danger"
                else:
                    bg = "#f0f0f0"  # Light gray for neutral
                    text_color = "text-dark"
            else:
                # Fallback to historical data with explicit checks
                if pos_pct >= 70:
                    bg = "#36ff00"  # Medium green
                    text_color = "text-success"
                elif pos_pct >= 60:
                    bg = "#70d800"  # Light green
                    text_color = "text-success"
                elif neg_pct >= 70:
                    bg = "#ff0000"  # Bright red
                    text_color = "text-danger"
                elif neg_pct >= 60:
                    bg = "#ff3333"  # Medium red
                    text_color = "text-danger"
                else:
                    bg = "#f0f0f0" if median >= 0 else "#ffcccc"  # Neutral or very light red
                    text_color = "text-dark" if median >= 0 else "text-danger"
        # Extra information to display
        extra_info = []
        if "prediction_accuracy" in day_info and not np.isnan(day_info["prediction_accuracy"]):
            extra_info.append(f"Acc: {day_info['prediction_accuracy']:.2f}")
        if "prediction_confidence" in day_info and not np.isnan(day_info["prediction_confidence"]):
            extra_info.append(f"Conf: {day_info['prediction_confidence']:.2f}")

        cell_content = [
            html.Div(f"TD {day_info['trading_day']}", className="fw-bold custom-day-number"),
            html.Div(
                f"{pos_pct:.1f}%" if median >= 0 else f"{neg_pct:.1f}%",
                className="small custom-cell-text"
            ) if not np.isnan(median) else html.Div("N/A", className="small custom-cell-text"),
            html.Div(f"{median:.2f}%", className=f"small {text_color} custom-cell-text"),
        ]

        if "pred_prob" in day_info and not np.isnan(day_info.get("pred_prob", np.nan)):
            direction = "↑" if day_info["pred_prob"] > 0.5 else "↓"
            cell_content.append(html.Div(
                f"{direction} {day_info['pred_prob']*100:.1f}%",
                className=f"small {'text-success' if day_info['pred_prob'] > 0.5 else 'text-danger'} custom-cell-text"))

        if extra_info:
            cell_content.append(html.Div(", ".join(extra_info), className="small text-info custom-cell-text"))
        cell_content.append(html.Div(f"n={day_info['sample_size']}", className="small text-muted custom-cell-text"))

        cell = html.Td(cell_content, className="p-2 text-center", style={"backgroundColor": bg})
        cells.append(cell)

    cell_rows = [html.Tr(cells[i:i+columns]) for i in range(0, len(cells), columns)]
    rows.extend(cell_rows)
    table = dbc.Table(rows, bordered=True, hover=True, responsive=True, striped=True, className="custom-table")

    legend_items = [
        html.Div([
            html.Span("Color Legend:", className="fw-bold me-2"),
            html.Span("Darker Green", className="px-2 py-1 me-1", style={"backgroundColor": "#00ff00", "color": "black"}),
            html.Span("= Strong Up Signal", className="me-3"),
            html.Span("Darker Red", className="px-2 py-1 me-1", style={"backgroundColor": "#ff0000", "color": "white"}),
            html.Span("= Strong Down Signal", className="me-3"),
            html.Span("Light Gray", className="px-2 py-1 me-1", style={"backgroundColor": "#f0f0f0", "color": "black"}),
            html.Span("= Neutral/Uncertain", className="me-3"),
        ], className="d-flex flex-wrap align-items-center justify-content-center mb-2")
    ]

    header_title = html.H3(f"{calendar.month_name[month]} Trading Days - SPY Seasonality", className="text-center my-3 fw-bold custom-table-header")
    return html.Div([header_title, *legend_items, table])
# -----------------------------
# Layout Definition
# -----------------------------

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Calendar Controls", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("◄ Previous", id="prev-month", color="primary", className="w-100")),
                        dbc.Col(dbc.Button("Next ►", id="next-month", color="primary", className="w-100"))
                    ], className="mb-3"),

                    html.Div(id="current-date-display", className="text-center my-2 fw-bold"),

                    html.Label("Historical Range:", className="mt-3"),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=2000,
                        max=2024,
                        step=1,
                        marks={i: str(i) for i in range(2000, 2025, 5)},
                        value=[2010, 2024],
                        tooltip={"placement": "bottom"}
                    ),
                    html.Div(id="year-range-display", className="text-center"),

                    html.Label("Minimum Sample Size:", className="mt-3"),
                    dcc.Slider(
                        id="min-sample-slider",
                        min=5,
                        max=20,
                        step=1,
                        marks={i: str(i) for i in [5, 10, 15, 20]},
                        value=10,
                        tooltip={"placement": "bottom"}
                    ),
                    html.Div(id="sample-size-display", className="text-center"),
                    dbc.Button("Apply Filters", id="apply-filters", color="success", className="w-100 mt-3")
                ])
            ], className="mb-4")
        ], width=3),

        dbc.Col([
            dbc.Alert("Loading data...", id="loading-alert", color="info", is_open=True),
            dbc.Spinner(html.Div(id="calendar-container"), color="primary"),
        ], width=9)

    ]),
    dcc.Interval(id="initial-load-interval", interval=1000, n_intervals=0, max_intervals=1),
    dcc.Store(id="current-date", data={"year": datetime.datetime.now().year, "month": datetime.datetime.now().month}),
    dcc.Store(id="seasonality-data", data=None),
    dcc.Store(id="model-status", data="uninitialized")
], fluid=True)

# -----------------------------
# Callbacks
# -----------------------------
from dash.dependencies import Input, Output, State

@dash.callback(
    Output("model-status", "data", allow_duplicate=True),
    Input("initial-load-interval", "n_intervals"),
    allow_duplicate=True,
    prevent_initial_call=True
)
def initialize_data(n_intervals):
    logger.info("Initial load interval triggered")
    return "ready"

@dash.callback(
    [Output("seasonality-data", "data"),
     Output("model-status", "data", allow_duplicate=True),
     Output("loading-alert", "children"),
     Output("loading-alert", "color"),
     Output("loading-alert", "is_open")],
    [Input("apply-filters", "n_clicks"),
     Input("model-status", "data")],
    [State("year-range-slider", "value"),
     State("min-sample-slider", "value")],
    prevent_initial_call=True
)
def update_seasonality(n_clicks, model_status, year_range, min_sample):
    try:
        spy_data = get_spy_data(f"{year_range[0]}-01-01", f"{year_range[1] + 1}-01-01")
        if spy_data.empty:
            return {}, "failed", "Failed to load data", "danger", True

        filtered = filter_correction_periods(spy_data)
        if len(filtered) < 252:
            return {}, "failed", "Insufficient data after filtering", "warning", True

        model, features = train_predictive_model(filtered)
        if model is None:
            return {}, "failed", "Model training failed", "danger", True

        filtered = add_predictions(filtered, model, features)
        seasonality = calculate_seasonality_with_predictions(filtered)

        return seasonality.to_dict('records'), "ready", "Data loaded successfully", "success", False
    except Exception as e:
        logger.error(f"Data update failed: {str(e)}")
        return {}, "failed", f"Error: {str(e)}", "danger", True

@dash.callback(
    Output("calendar-container", "children"),
    Output("current-date-display", "children"),
    Input("current-date", "data"),
    Input("min-sample-slider", "value"),
    Input("seasonality-data", "data")
)
def update_calendar(current_date, min_sample, seasonality_json):
    trigger = dash.callback_context.triggered[0]["prop_id"]
    logger.info(f"update_calendar called with trigger: {trigger}")

    if not seasonality_json:
        logger.warning("No seasonality data available")
        return html.Div(html.H4("No seasonality data available. Please click 'Apply Filters' to load data.",
                              className="text-center text-warning my-5")), ""

    try:
        seasonality = pd.DataFrame(seasonality_json)
        logger.info(f"Creating calendar for {current_date['month']}/{current_date['year']} with {len(seasonality)} records")
        cal_layout = create_calendar_layout(current_date["year"], current_date["month"], seasonality, min_sample)
        date_str = f"Viewing: {calendar.month_name[current_date['month']]} {current_date['year']}"
        return cal_layout, date_str
    except Exception as e:
        logger.error(f"Error updating calendar: {str(e)}", exc_info=True)
        return html.Div(f"Error creating calendar: {str(e)}"), ""

@dash.callback(
    Output("year-range-display", "children"),
    Input("year-range-slider", "value")
)
def update_year_range_display(year_range):
    return f"Data from {year_range[0]} to {year_range[1]}"

@dash.callback(
    Output("sample-size-display", "children"),
    Input("min-sample-slider", "value")
)
def update_sample_size_display(min_sample):
    return f"Minimum sample size: {min_sample}"

@dash.callback(
    Output("current-date", "data", allow_duplicate=True),
    Input("prev-month", "n_clicks"),
    State("current-date", "data"),
    prevent_initial_call=True
)
def previous_month(n_clicks, current_date):
    if n_clicks:
        logger.info(f"Moving to previous month from {current_date['month']}/{current_date['year']}")
        current = datetime.date(current_date["year"], current_date["month"], 1)
        prev_mon = current - relativedelta(months=1)
        return {"year": prev_mon.year, "month": prev_mon.month}
    return current_date

@dash.callback(
    Output("current-date", "data", allow_duplicate=True),
    Input("next-month", "n_clicks"),
    State("current-date", "data"),
    prevent_initial_call=True
)
def next_month(n_clicks, current_date):
    if n_clicks:
        logger.info(f"Moving to next month from {current_date['month']}/{current_date['year']}")
        current = datetime.date(current_date["year"], current_date["month"], 1)
        next_mon = current + relativedelta(months=1)
        return {"year": next_mon.year, "month": next_mon.month}
    return current_date
