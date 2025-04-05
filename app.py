import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# We import dash.pages but don't necessarily need it explicitly if we're just using use_pages=True
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.MINTY],  # Choose a modern, bright theme
    suppress_callback_exceptions=True
)

# Expose server for deployment
server = app.server

# Define a modern navbar with brand and page links
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Pog Trading", className="ms-2", style={"fontSize": "1.5rem", "fontWeight": "bold"}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Calendar", href="/calendar", active="exact")),
        ], className="ms-auto", navbar=True),
    ]),
    color="primary",
    dark=True,
    className="mb-4 shadow-sm"
)

app.layout = html.Div([
    navbar,
    # This is where pages will be displayed
    dash.page_container
])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
