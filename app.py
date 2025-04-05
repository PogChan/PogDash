import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash.pages  # Enables multi-page support

# Initialize the Dash app using a Bootstrap theme (Cyborg gives a modern, dark look)
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # For Render deployment

# Layout with a navigation bar and a page container for dynamic content
app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dcc.Link("Calendar", href="/calendar", className="nav-link")),
            # Additional pages (like "Statistics", "About", etc.) can be added here.
        ],
        brand="Pog Trading",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    dash.page_container
], fluid=True)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8080, debug=True)
