import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Initialize Dash with the CYBORG theme (a dark base) plus our custom CSS in assets/style.css
app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)
server = app.server

# A modern navigation bar with links to Home and Calendar pages
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Pog Trading", className="ms-2 custom-brand"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Calendar", href="/calendar", active="exact")),
        ], className="ms-auto", navbar=True),
    ]),
    color="dark",
    dark=True,
    className="mb-4 shadow-sm"
)

app.layout = html.Div([
    navbar,
    dash.page_container
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
