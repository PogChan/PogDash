import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Register Home page with the multi‑page system
dash.register_page(__name__, path="/", name="Home")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Welcome to Pog Trading", className="display-3 fw-bold custom-title"),
                html.P(
                    "Discover market insights and SPY seasonality trends with our sleek, modern interface.",
                    className="lead custom-text"
                ),
                dbc.Button(
                    "View Calendar",
                    href="/calendar",
                    color="warning",
                    className="mt-3 btn-lg shadow-sm custom-button"
                )
            ], className="hero-section text-center p-5")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Bullish/Bearish Insights", className="card-title fw-bold custom-card-title"),
                    html.P(
                        "Our SPY calendar helps you identify days with a high probability of bullish or bearish movements.",
                        className="custom-text"
                    )
                ])
            ], className="mb-4 shadow-sm custom-card")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Intuitive Navigation", className="card-title fw-bold custom-card-title"),
                    html.P(
                        "Built on a modern, responsive design, navigate easily between pages and access real-time insights.",
                        className="custom-text"
                    )
                ])
            ], className="mb-4 shadow-sm custom-card")
        ], width=6),
    ], className="mb-5")
], fluid=True)
