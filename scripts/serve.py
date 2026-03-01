# /// script
# requires-python = ">=3.10"
# dependencies = ["RangeHTTPServer"]
# ///
"""CORS + Range HTTP server for viewing zarr stores in neuroglancer.

Usage:
    uv run scripts/serve.py <zarr_path> [--port PORT] [--no-open]

Opens neuroglancer in your browser pointing at the zarr store.

Examples:
    uv run scripts/serve.py build/visual.zarr
    uv run scripts/serve.py build/visual.zarr --port 9000
"""

import argparse
import json
import os
import webbrowser
from http.server import HTTPServer
from pathlib import Path
from urllib.parse import quote

from RangeHTTPServer import RangeRequestHandler


class Handler(RangeRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Range, Content-Length"
        )
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def read_axes(zarr_path):
    """Read axis names and types from OME-NGFF metadata."""
    meta_path = zarr_path / "zarr.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    try:
        return meta["attributes"]["ome"]["multiscales"][0]["axes"]
    except (KeyError, IndexError):
        return None


def neuroglancer_url(port, zarr_rel_path, axes=None):
    source = f"zarr://http://localhost:{port}/{zarr_rel_path}"
    state = {
        "layers": [{"type": "image", "source": source, "name": "data"}],
    }

    if axes:
        # Show spatial dims (z, y, x) in the cross-section panels
        spatial = [a["name"] for a in axes if a.get("type") == "space"]
        if spatial:
            state["displayDimensions"] = spatial

    state_json = json.dumps(state, separators=(",", ":"))
    return f"https://neuroglancer-demo.appspot.com/#!{quote(state_json)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to zarr store")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path).resolve()
    serve_dir = zarr_path.parent
    zarr_rel = zarr_path.name

    axes = read_axes(zarr_path)

    os.chdir(serve_dir)

    url = neuroglancer_url(args.port, zarr_rel, axes)
    print(f"Serving {serve_dir} on http://localhost:{args.port}")
    print(f"Neuroglancer: {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        webbrowser.open(url)

    server = HTTPServer(("", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
