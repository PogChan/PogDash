import dash
from dash import html
import dash_bootstrap_components as dbc
import pages.calendar

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="SPY Seasonality Dashboard",
        color="primary",
        dark=True,
    ),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True)