import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Register the page
dash.register_page(__name__, path="/", name="Home")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Welcome to Pog Trading", className="display-3 fw-bold text-primary"),
                html.P(
                    "Discover market insights and SPY seasonality trends in a clean, intuitive interface.",
                    className="lead"
                ),
                dbc.Button(
                    "View Calendar",
                    href="/calendar",
                    color="success",
                    className="mt-3 btn-lg shadow-sm"
                )
            ], className="hero-section text-center p-5")
        ], width=12)
    ]),

    # Example row for additional info or highlights
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Bullish/Bearish Insights", className="card-title fw-bold text-success"),
                    html.P(
                        "Our SPY calendar helps you identify which days have historically trended green or red. "
                        "Use this info to plan your trades and manage risk effectively."
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Easy to Navigate", className="card-title fw-bold text-info"),
                    html.P(
                        "Built with Dash's multi-page architecture, so you can quickly move between pages. "
                        "More features coming soon!"
                    )
                ])
            ], className="mb-4 shadow-sm")
        ], width=6),
    ], className="mb-5"),
], fluid=True)
