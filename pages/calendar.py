import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import calendar
from dateutil.relativedelta import relativedelta
import calendar

dash.register_page(__name__, path="/calendar", name="Calendar")

# -----------------------------
# Data Functions
# -----------------------------

def get_spy_data(start_date="2000-01-01"):
    data = yf.download("SPY", start=start_date)
    # Calculate daily returns using the unadjusted "Close"
    data["Daily_Return"] = data["Close"].pct_change() * 100
    return data

def filter_correction_periods(data, threshold=-15):
    """
    For each year-month group, if the minimum rolling 30-day return
    in that month is below the threshold, exclude that entire month.
    """
    data["Rolling_30_Return"] = data["Close"].pct_change(30) * 100

    def filter_month(group):
        if group["Rolling_30_Return"].min() < threshold:
            return pd.DataFrame()  # drop the whole month
        return group

    filtered = data.groupby([data.index.year, data.index.month], group_keys=False).apply(filter_month)
    return filtered

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

    return stats

def generate_calendar_data(seasonality, month):
    month_data = seasonality[seasonality["Month"] == month]
    if month_data.empty:
        max_trading_day = 0
    else:
        max_trading_day = int(month_data["Trading_Day"].max())
    calendar_list = []
    for td in range(1, max_trading_day + 1):
        row = month_data[month_data["Trading_Day"] == td]
        if not row.empty:
            median = row["median"].values[0]
            pos_pct = row["pos_pct"].values[0]
            neg_pct = row["neg_pct"].values[0]
            sample_size = int(row["count"].values[0])
        else:
            median = np.nan
            pos_pct = np.nan
            neg_pct = np.nan
            sample_size = 0
        calendar_list.append({
            "trading_day": td,
            "median": median,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
            "sample_size": sample_size
        })
    return calendar_list

def create_calendar_layout(year, month, seasonality_df, min_sample=10, columns=5):
    # Generate trading day data for the month
    cal_data = generate_calendar_data(seasonality_df, month)

    # Create header row with trading day labels
    header = html.Tr([html.Th(f"TD {i+1}", className="text-center custom-table-header") for i in range(columns)])
    rows = [header]

    cells = []
    for day_info in cal_data:
        if day_info["sample_size"] < min_sample or np.isnan(day_info["pos_pct"]):
            bg = "#1a1a1a"  # Insufficient data: dark background
            text_color = "text-muted"
        else:
            median = day_info["median"]
            pos_pct = day_info["pos_pct"]
            neg_pct = day_info["neg_pct"]
            # If median is positive, use green shades
            if median >= 0:
                if pos_pct >= 70:
                    bg = "#36ff00"  # Bright green
                elif pos_pct >= 60:
                    bg = "#70d800"  # Moderately bright green
                else:
                    bg = "#333333"  # Neutral dark gray for middling bullish probability
                text_color = "text-success"
            # If median is negative, use red shades
            else:
                if neg_pct >= 70:
                    bg = "#ff0000"  # Bright red
                    text_color = "text-danger"
                elif neg_pct >= 60:
                    bg = "#ff6666"  # Moderately bright red
                    text_color = "text-danger"
                else:
                    bg = "#333333"  # Neutral dark gray for low bearish probability
                    text_color = "text-muted"

        cell_content = [
            html.Div(f"TD {day_info['trading_day']}", className="fw-bold custom-day-number"),
            html.Div(
                f"{day_info['pos_pct']:.1f}%" if day_info["median"] >= 0 else f"{day_info['neg_pct']:.1f}%",
                className="small custom-cell-text"
            ) if not np.isnan(day_info["median"]) else html.Div("N/A", className="small custom-cell-text"),
            html.Div(f"{day_info['median']:.2f}%" if not np.isnan(day_info.get("median", np.nan)) else "N/A", className=f"small {text_color} custom-cell-text"),
            html.Div(f"n={day_info['sample_size']}", className="small text-muted custom-cell-text")
        ]
        cell = html.Td(cell_content, className="p-2 text-center", style={"backgroundColor": bg})
        cells.append(cell)

    # Group cells into rows with the specified number of columns
    cell_rows = [html.Tr(cells[i:i+columns]) for i in range(0, len(cells), columns)]
    rows.extend(cell_rows)

    table = dbc.Table(rows, bordered=True, hover=True, responsive=True, striped=True, className="custom-table")
    header_title = html.H3(f"{calendar.month_name[month]} Trading Days - SPY Seasonality", className="text-center my-3 fw-bold custom-table-header")
    return html.Div([header_title, table])

# -----------------------------
# Layout Definition
# -----------------------------

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Calendar Navigation", className="fw-bold text-white custom-card-header"),
                dbc.CardBody([
                    dbc.Button("Previous Month", id="prev-month", color="warning", className="me-2 custom-button"),
                    dbc.Button("Next Month", id="next-month", color="warning", className="custom-button"),
                    html.Div(id="current-date-display", className="mt-3 text-center custom-text"),

                    html.Label("Year Range:", className="mt-3 fw-bold custom-text"),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=2000,
                        max=2024,
                        step=1,
                        marks={i: str(i) for i in range(2000, 2025, 5)},
                        value=[2010, 2024]
                    ),
                    html.Div(id="year-range-display", className="mt-2 custom-text"),

                    html.Label("Minimum Sample Size:", className="mt-3 fw-bold custom-text"),
                    dcc.Slider(
                        id="min-sample-slider",
                        min=5,
                        max=20,
                        step=1,
                        marks={i: str(i) for i in [5, 10, 15, 20]},
                        value=10
                    ),
                    html.Div(id="sample-size-display", className="mt-2 custom-text"),

                    dbc.Button("Apply Filters", id="apply-filters", color="warning", className="w-100 mt-3 fw-bold custom-button"),
                ])
            ], className="mb-4 shadow-sm custom-card"),
        ], width=3),

        dbc.Col([
            dbc.Spinner(html.Div(id="calendar-container", className="p-3"), color="warning")
        ], width=9)
    ]),

    dcc.Store(id="current-date", data={"year": datetime.datetime.now().year, "month": datetime.datetime.now().month}),
    dcc.Store(id="seasonality-data", data=[]),
    dcc.Interval(
        id="initial-load",
        interval=1*1000,  # fires once after 1 second
        n_intervals=0,
        max_intervals=1
    ),


    html.Hr(),
    html.Footer([
        html.P("SPY Seasonality Calendar © 2025", className="text-center mb-0 custom-text"),
        html.P("Data source: Yahoo Finance via yfinance", className="text-center mb-0 custom-text"),
        html.P("Excludes entire month if a correction (<-15%) is detected", className="text-center custom-text"),
    ], className="my-3")
], fluid=True)

# -----------------------------
# Callbacks
# -----------------------------
from dash.dependencies import Input, Output, State

@dash.callback(
    Output("seasonality-data", "data"),
    [Input("apply-filters", "n_clicks"),
     Input("initial-load", "n_intervals")],
    [State("year-range-slider", "value"),
     State("min-sample-slider", "value"),
     State("current-date", "data")],
    prevent_initial_call=False
)
def update_seasonality(n_clicks, n_intervals, year_range, min_sample, current_date):
    start_date = f"{year_range[0]}-01-01"
    spy_data = get_spy_data(start_date)
    filtered = filter_correction_periods(spy_data)
    seasonality = calculate_seasonality(filtered)
    return seasonality.to_dict("records")

@dash.callback(
    Output("calendar-container", "children"),
    Output("current-date-display", "children"),
    Input("current-date", "data"),
    Input("min-sample-slider", "value"),
    State("seasonality-data", "data")
)
def update_calendar(current_date, min_sample, seasonality_json):
    if not seasonality_json:
        return html.Div("No seasonality data available."), ""
    seasonality = pd.DataFrame(seasonality_json)
    cal_layout = create_calendar_layout(current_date["year"], current_date["month"], seasonality, min_sample)
    date_str = f"Viewing: Month {current_date['month']} of {current_date['year']}"
    return cal_layout, date_str

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
        current = datetime.date(current_date["year"], current_date["month"], 1)
        next_mon = current + relativedelta(months=1)
        return {"year": next_mon.year, "month": next_mon.month}
    return current_date
