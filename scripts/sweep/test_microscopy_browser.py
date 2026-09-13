# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.62.0"]
# ///
"""Check microscopy plot scales, selection, and retained data in Chromium."""
import argparse
import csv
from functools import partial
from http.server import ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading

from site_server import ReportHandler


class QuietHandler(ReportHandler):
    def log_message(self, *args):
        pass


def plot_points(page):
    return page.locator(".study-plot").evaluate_all("""panels => panels.map(panel =>
      Array.from(panel.querySelectorAll('.point'), point => {
        const matrix = point.transform.baseVal.consolidate().matrix;
        return {id: point.dataset.id, x: matrix.e, y: matrix.f,
          description: point.querySelector('title').textContent,
          label: point.getAttribute('aria-label')};
      }))""")


def check_axes(page, rows):
    by_id = {row["id"]: row for row in rows}
    for points in plot_points(page):
        assert points
        folds = [(point, by_id[point["id"]]["compression_fold"]) for point in points]
        rates = [(point, by_id[point["id"]]["throughput"]["median"]) for point in points]
        for values, coordinate, transform, direction in [(folds, "x", math.log, 1), (rates, "y", float, -1)]:
            lo, hi = min(values, key=lambda item: item[1]), max(values, key=lambda item: item[1])
            if lo[1] == hi[1]:
                continue
            slope = (hi[0][coordinate] - lo[0][coordinate]) / (transform(hi[1]) - transform(lo[1]))
            assert slope * direction > 0
            for point, value in values:
                expected = lo[0][coordinate] + slope * (transform(value) - transform(lo[1]))
                assert math.isclose(point[coordinate], expected, abs_tol=1e-4)
        for point in points:
            row = by_id[point["id"]]
            description = point["description"]
            assert point["label"] == description
            assert row["config"]["backend"].upper() in description
            assert row["config"]["sink"] in description
            assert row["input_label"] in description
            assert " × ".join(map(str, row["detail"]["image_replay"]["chunk_shape"])) in description
            assert "Logical compression fold" in description
            assert "observation(s)" in description
            assert "workers" in description
            if row.get("uncertainty"):
                assert "round resamples" in description
                assert "Observed fold" in description
            if row["count"] == 1:
                assert "variation unknown" in description.lower()
            if row["config"]["codec"].startswith("blosc-"):
                assert "bitshuffle" in description and "Blosc block request" in description
    assert page.locator(".fold-range").count() == sum(
        row["count"] > 1 and "compression_range" in row for row in rows)
    assert page.locator(".fold-range").evaluate_all(
        "lines => lines.every(line => line.getAttribute('y1') === line.getAttribute('y2'))")
    frontier_ids = set(page.locator(".point.frontier").evaluate_all("points => points.map(point => point.dataset.id)"))
    supported = {row["id"] for row in rows if row.get("uncertainty", {}).get("frontier_frequency", 0) >= 0.05}
    assert page.locator(".frontier-ring").count() == len(supported - frontier_ids)
    for svg in page.locator(".study-plot > svg").all():
        assert svg.locator(".axis-title").all_text_contents() == [
            "Logical compression fold (×)", "Logical throughput (GiB/s)"]
    assert page.locator(".range").evaluate_all(
        "lines => lines.every(line => line.getAttribute('x1') === line.getAttribute('x2'))")


def set_theme(page, theme):
    for _ in range(3):
        if page.locator("html").get_attribute("data-theme") == theme:
            return
        page.locator("#theme-toggle").click()
    assert page.locator("html").get_attribute("data-theme") == theme


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
                check_axes(page, rows)
                expect(page.locator("#axis-note")).to_contain_text("limits differ")
                expect(page.locator("#axis-note")).to_contain_text("fold below 1")
                options = page.locator("#codec option").all_text_contents()
                assert {"blosc-lz4", "blosc-zstd", "lz4 (raw)", "zstd (raw)"} <= set(options)
                uncertain = next((row for row in rows if row.get("uncertainty")), None)
                if uncertain:
                    page.locator(f'.point[data-id="{uncertain["id"]}"]').click()
                    expect(page.locator("#detail-content")).to_contain_text("Variation across rounds")
                    expect(page.locator("#detail-content")).to_contain_text("Approximate 95% bootstrap intervals")
                    expect(page.locator("#detail-content")).to_contain_text("including hidden settings")
                    expect(page.locator("#axis-note")).to_contain_text("all selected settings")
                page.locator(".point.frontier").first.click()
                expect(page.locator("#detail-content")).to_contain_text("Pipeline stages")
                plots_box = page.locator("#plots").bounding_box()
                detail_box = page.locator("#detail").bounding_box()
                assert detail_box["x"] >= plots_box["x"] + plots_box["width"]
                assert abs(detail_box["y"] - plots_box["y"]) <= 1
                page.evaluate("top => scrollTo(0, top + 50)", plots_box["y"])
                detail_box = page.locator("#detail").bounding_box()
                assert 0 <= detail_box["y"] <= 20
                assert detail_box["y"] + detail_box["height"] <= 1050
                page.evaluate("scrollTo(0, 0)")
                frontier_count = page.locator(".point.frontier").count()
                page.locator("#fit-frontier").click()
                expect(page.locator("#fit-frontier")).to_have_attribute("aria-pressed", "true")
                expect(page.locator("#axis-note")).to_contain_text("outside the plot limits")
                expect(page.locator("#table-body tr")).to_have_count(len(rows))
                assert page.locator(".point").count() == len(rows)
                assert page.locator(".point.frontier").count() == frontier_count
                assert page.locator(".point.frontier[tabindex='-1']").count() == 0
                assert page.locator(".point.resampled-frontier[tabindex='-1']").count() == 0
                set_theme(page, "light")
                page.screenshot(path=str(screenshots / "frontier-focus.png"), full_page=True)
                page.reload(wait_until="networkidle")
                expect(page.locator("#fit-frontier")).to_have_attribute("aria-pressed", "true")
                page.locator("#show-all").click()
                assert page.locator(".point[tabindex='-1']").count() == 0
                for mode in ["all", "input"]:
                    page.select_option("#axes", mode)
                    expect(page.locator("#table-body tr")).to_have_count(len(rows))
                    panels = page.locator(".study-plot").evaluate_all("""panels => panels.map(panel => ({
                      id: panel.querySelector('.point').dataset.id,
                      ticks: Array.from(panel.querySelectorAll('.plot-axis .tick text'), node => node.textContent)
                    }))""")
                    grouped = {}
                    by_id = {row["id"]: row for row in rows}
                    for panel in panels:
                        key = "all" if mode == "all" else by_id[panel["id"]]["config"]["input_id"]
                        if key in grouped:
                            assert panel["ticks"] == grouped[key]
                        grouped[key] = panel["ticks"]
                page.reload(wait_until="networkidle")
                expect(page.locator("#axes")).to_have_value("input")
                page.select_option("#axes", "panel")
                for codec in ["blosc-lz4", "lz4", "blosc-zstd", "zstd"]:
                    page.select_option("#codec", codec)
                    expected = [row for row in rows if row["config"]["codec"] == codec]
                    expect(page.locator("#table-body tr")).to_have_count(len(expected))
                    assert page.locator(".point").count() == len(expected)
                page.select_option("#codec", "all")
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
                page.select_option("#block", "all")
                expect(page.locator("#table-body tr")).to_have_count(len(rows))
                page.evaluate("scrollTo(0, 0)")
                set_theme(page, "light")
                page.screenshot(path=str(screenshots / "desktop-light.png"), full_page=True)
                set_theme(page, "dark")
                page.screenshot(path=str(screenshots / "desktop-dark.png"), full_page=True)
                set_theme(page, "light")
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
                raw_links = page.get_by_role("link", name="Retained raw observations", exact=True)
                assert raw_links.count() == 1
                for link in raw_links.all():
                    raw_url = link.get_attribute("href")
                    response = context.request.get(f"http://127.0.0.1:{server.server_port}/{raw_url}")
                    assert response.ok
                for data in datasets:
                    response = context.request.get(f"http://127.0.0.1:{server.server_port}/{data['study']['archive']}")
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
