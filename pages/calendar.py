import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import calendar
from dateutil.relativedelta import relativedelta

dash.register_page(__name__, path="/calendar", name="Calendar")

def get_spy_data(start_date="2000-01-01"):
    data = yf.download("SPY", start=start_date)
    # Use "Adj Close" if available; otherwise, use "Close"
    data["Daily_Return"] = data["Close"].pct_change() * 100
    return data

def filter_correction_periods(data, threshold=-15):
    data["Rolling_30_Return"] = data["Close"].pct_change(30) * 100
    correction_starts = data[data["Rolling_30_Return"] < threshold].index
    exclude_mask = pd.Series(False, index=data.index)
    for start in correction_starts:
        end = start + pd.DateOffset(months=3)
        exclude_mask |= (data.index >= start) & (data.index <= end)
    return data[~exclude_mask]

def calculate_seasonality(data):
    data = data.copy()
    data["Month"] = data.index.month
    data["Day"] = data.index.day

    stats = data.groupby(["Month", "Day"])["Daily_Return"].agg(["mean", "count"]).reset_index()

    pos_counts = data[data["Daily_Return"] > 0].groupby(["Month", "Day"]).size().reset_index(name="pos_count")
    stats = pd.merge(stats, pos_counts, on=["Month", "Day"], how="left")
    stats["pos_count"] = stats["pos_count"].fillna(0)
    stats["pos_pct"] = (stats["pos_count"] / stats["count"] * 100).round(1)
    return stats

def generate_calendar_data(seasonality, year, month):
    num_days = calendar.monthrange(year, month)[1]
    start_date = datetime.date(year, month, 1)
    calendar_list = []
    for day in range(1, num_days + 1):
        current_date = datetime.date(year, month, day)
        weekday = current_date.weekday()  # Monday=0 ... Sunday=6
        row = seasonality[(seasonality["Month"] == month) & (seasonality["Day"] == day)]
        if not row.empty:
            avg_return = row["mean"].values[0]
            pos_pct = row["pos_pct"].values[0]
            sample_size = int(row["count"].values[0])
        else:
            avg_return = np.nan
            pos_pct = np.nan
            sample_size = 0
        calendar_list.append({
            "date": current_date,
            "day": day,
            "weekday": weekday,
            "avg_return": avg_return,
            "pos_pct": pos_pct,
            "sample_size": sample_size
        })
    return calendar_list

def create_calendar_layout(year, month, seasonality_df, min_sample=10):
    cal_data = generate_calendar_data(seasonality_df, year, month)
    first_weekday = datetime.date(year, month, 1).weekday()
    # Sunday as first column => offset
    first_day_offset = (first_weekday + 1) % 7

    weekdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    header = html.Tr([html.Th(day, className="text-center") for day in weekdays])
    rows = [header]

    day_idx = 0
    weeks = 6
    for _ in range(weeks):
        cells = []
        for day in range(7):
            if _ == 0 and day < first_day_offset:
                cells.append(html.Td("", className="p-2"))
            elif day_idx < len(cal_data):
                day_info = cal_data[day_idx]
                day_idx += 1
                if day_info["sample_size"] < min_sample or np.isnan(day_info["pos_pct"]):
                    bg = "#f8f9fa"
                    text_color = "text-muted"
                else:
                    pos = day_info["pos_pct"]
                    if pos >= 70:
                        bg = "#00cc00"
                    elif pos >= 60:
                        bg = "#99ff99"
                    elif pos >= 40:
                        bg = "#f2f2f2"
                    elif pos >= 30:
                        bg = "#ffcccc"
                    else:
                        bg = "#ff6666"

                    if np.isnan(day_info["avg_return"]):
                        text_color = "text-muted"
                    elif day_info["avg_return"] > 0:
                        text_color = "text-success"
                    else:
                        text_color = "text-danger"

                cell_content = [
                    html.Div(day_info["day"], className="fw-bold"),
                    html.Div(f"{day_info['pos_pct']:.1f}%" if not np.isnan(day_info["pos_pct"]) else "N/A", className="small"),
                    html.Div(f"{day_info['avg_return']:.2f}%" if not np.isnan(day_info["avg_return"]) else "N/A", className=f"small {text_color}"),
                    html.Div(f"n={day_info['sample_size']}", className="small text-muted")
                ]
                cells.append(html.Td(cell_content, className="p-2 text-center", style={"backgroundColor": bg}))
            else:
                cells.append(html.Td("", className="p-2"))
        rows.append(html.Tr(cells))

    table = dbc.Table(rows, bordered=True, hover=True, responsive=True, striped=True)
    header_title = html.H3(f"{calendar.month_name[month]} {year} - SPY Seasonality", className="text-center my-3 fw-bold")
    return html.Div([header_title, table])

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Calendar Navigation", className="fw-bold text-white bg-primary"),
                dbc.CardBody([
                    dbc.Button("Previous Month", id="prev-month", color="secondary", className="me-2"),
                    dbc.Button("Next Month", id="next-month", color="secondary"),
                    html.Div(id="current-date-display", className="mt-3 text-center"),

                    html.Label("Year Range:", className="mt-3 fw-bold"),
                    dcc.RangeSlider(
                        id="year-range-slider",
                        min=2000,
                        max=2024,
                        step=1,
                        marks={i: str(i) for i in range(2000, 2025, 5)},
                        value=[2010, 2024]
                    ),
                    html.Div(id="year-range-display", className="mt-2"),

                    html.Label("Minimum Sample Size:", className="mt-3 fw-bold"),
                    dcc.Slider(
                        id="min-sample-slider",
                        min=5,
                        max=20,
                        step=1,
                        marks={i: str(i) for i in [5, 10, 15, 20]},
                        value=10
                    ),
                    html.Div(id="sample-size-display", className="mt-2"),

                    dbc.Button("Apply Filters", id="apply-filters", color="success", className="w-100 mt-3 fw-bold"),
                ])
            ], className="mb-4 shadow-sm"),
        ], width=3),

        dbc.Col([
            dbc.Spinner(html.Div(id="calendar-container", className="p-3"), color="primary")
        ], width=9)
    ]),

    dcc.Store(id="seasonality-data"),
    dcc.Store(id="current-date", data={"year": datetime.datetime.now().year, "month": datetime.datetime.now().month}),

    html.Hr(),
    html.Footer([
        html.P("SPY Seasonality Calendar © 2025", className="text-center mb-0"),
        html.P("Data source: Yahoo Finance via yfinance", className="text-center mb-0"),
        html.P("Excludes major correction periods (>15% drops in 30 days)", className="text-center"),
    ], className="my-3"),
], fluid=True)

from dash.dependencies import Input, Output, State

@dash.callback(
    Output("seasonality-data", "data"),
    Input("apply-filters", "n_clicks"),
    State("year-range-slider", "value"),
    State("min-sample-slider", "value"),
    State("current-date", "data"),
    prevent_initial_call=False
)
def update_seasonality(n_clicks, year_range, min_sample, current_date):
    start_date = f"{year_range[0]}-01-01"
    spy_data = get_spy_data(start_date)
    filtered = filter_correction_periods(spy_data)
    seasonality = calculate_seasonality(filtered)
    return seasonality.to_dict("records")

def initialize_data(n_clicks, year_range, min_sample, current_date):
    start_date = f"{year_range[0]}-01-01"
    spy_data = get_spy_data(start_date)
    filtered = filter_correction_periods(spy_data)
    seasonality = calculate_seasonality(filtered)
    seasonality_json = seasonality.to_dict("records")
    cal_layout = create_calendar_layout(current_date["year"], current_date["month"], seasonality, min_sample)
    return seasonality_json, cal_layout

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
    date_str = f"Viewing: {calendar.month_name[current_date['month']]} {current_date['year']}"
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
