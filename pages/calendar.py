import dash
from dash import html, dcc
import pandas as pd
import yfinance as yf

dash.register_page(__name__, path="/calendar", name="📅 Calendar")

def get_calendar_data():
    spy = yf.download('SPY', start='2004-01-01', end='2024-12-31')
    spy['Date'] = spy.index
    spy['Green'] = spy['Close'] > spy['Open']
    spy['Day'] = spy['Date'].dt.day
    spy['Month'] = spy['Date'].dt.month
    spy['Year'] = spy['Date'].dt.year

    correction_months = [(2020, 3), (2018, 12), (2008, 10)]
    spy = spy[~spy[['Year', 'Month']].apply(tuple, axis=1).isin(correction_months)]

    stats = spy.groupby('Day')['Green'].agg(['count', 'sum']).reset_index()
    stats.columns = ['Day', 'Total', 'GreenDays']
    stats['%Green'] = (stats['GreenDays'] / stats['Total'] * 100).round(1)
    stats['%Red'] = 100 - stats['%Green']
    return stats

df = get_calendar_data()

layout = html.Div([
    html.H2("📈 SPY Green/Red Days by Calendar Day"),
    dcc.Graph(
        figure={
            "data": [{
                "x": df["Day"],
                "y": df["%Green"],
                "type": "bar",
                "marker": {"color": "green"},
                "name": "% Green Days"
            }, {
                "x": df["Day"],
                "y": df["%Red"],
                "type": "bar",
                "marker": {"color": "red"},
                "name": "% Red Days"
            }],
            "layout": {
                "barmode": "stack",
                "title": "Historical % Green/Red SPY Days (Excludes Major Corrections)",
                "xaxis": {"title": "Day of Month"},
                "yaxis": {"title": "Percentage"}
            }
        }
    )
])