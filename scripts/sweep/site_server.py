"""Static report handler, independent of platform MIME registry settings."""
from http.server import SimpleHTTPRequestHandler


class ReportHandler(SimpleHTTPRequestHandler):
    # Windows can register .js as text/plain, which browsers reject for modules.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript",
                      ".json": "application/json"}
