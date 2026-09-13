# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.62.0"]
# ///
"""Check a built microscopy report containing test fixtures in Chromium."""
import argparse
import csv
from functools import partial
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading

from site_server import ReportHandler


class QuietHandler(ReportHandler):
    def log_message(self, *args):
        pass


def check_site(site, screenshots, executable=None):
    from playwright.sync_api import expect, sync_playwright

    index = json.loads((site / "data/microscopy/index.json").read_text())
    datasets = [json.loads((site / item["file"]).read_text()) for item in index["studies"]]
    rows = [row for data in datasets for row in data["measurements"]]
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            options = {"executable_path": str(executable)} if executable else {}
            browser = playwright.chromium.launch(headless=True, **options)
            context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/microscopy.html?input=all", wait_until="networkidle")
            if not rows:
                expect(page.locator("#load-status")).to_contain_text("No retained microscopy study")
                expect(page.locator("#workspace")).to_be_hidden()
                assert page.get_by_role("link", name="Download the discovery definition").get_attribute("href")
                page.screenshot(path=str(screenshots / "empty.png"), full_page=True)
            else:
                expect(page.locator("#workspace")).to_be_visible()
                expect(page.locator("#table-body tr")).to_have_count(len(rows))
                assert page.locator(".point").count() == len(rows)
                page.select_option("#block", "16384")
                filtered = [row for row in rows if not row["config"]["codec"].startswith("blosc-")
                            or row["config"]["blosc_block_bytes"] == 16384]
                expect(page.locator("#table-body tr")).to_have_count(len(filtered))
                page.get_by_role("button", name="Raw LZ4", exact=True).first.click()
                expect(page.locator("#detail-content")).to_contain_text("Pipeline stages")
                expect(page.locator("#detail-content")).to_contain_text("compress")
                expect(page.locator("#detail-content")).to_contain_text("Raw LZ4")
                assert "selected=" in page.url
                url = page.url
                page.reload(wait_until="networkidle")
                expect(page.locator("#table-body tr")).to_have_count(len(filtered))
                expect(page.locator("#detail-content")).to_contain_text("Raw LZ4")
                assert page.url == url
                with page.expect_download() as download:
                    page.locator("#download").click()
                exported = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text())))
                assert len(exported) == len(filtered)
                assert {row["codec"] for row in exported} >= {"none", "lz4", "zstd"}
                page.select_option("#codec", "zstd")
                expect(page.locator("#detail-content")).to_contain_text("outside the current filters")
                page.go_back(wait_until="networkidle")
                expect(page.locator("#table-body tr")).to_have_count(len(filtered))
                page.screenshot(path=str(screenshots / "desktop-light.png"), full_page=True)
                page.locator("#theme-toggle").click()
                page.screenshot(path=str(screenshots / "desktop-dark.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(200)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                page.screenshot(path=str(screenshots / "mobile.png"), full_page=True)
                page.locator("#reset").click()
                expect(page.locator("#block")).to_have_value("all")
                assert page.locator(".point").count() > 0
                page.locator(".point").first.focus()
                page.keyboard.press("Enter")
                expect(page.locator("#detail-content")).to_contain_text("Pipeline stages")
                raw_url = page.get_by_role("link", name="Retained raw observations", exact=True).get_attribute("href")
                response = context.request.get(f"http://127.0.0.1:{server.server_port}/{raw_url}")
                assert response.ok
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    check_site(args.site, args.screenshots, args.executable)
    print("Microscopy browser checks passed")


if __name__ == "__main__":
    main()
