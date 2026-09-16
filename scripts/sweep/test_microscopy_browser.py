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
            assert "Spatial shard files" in description
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


def select_filter(page, key, value):
    if page.locator("#filter-panel").get_attribute("open") is None:
        page.locator("#filter-panel > summary").click()
    page.select_option(f"#{key}", value)


def select_input(page, input_id):
    button = page.locator(f'#dataset-tabs button[data-input="{input_id}"]')
    if button.is_visible():
        button.click()
    else:
        page.select_option("#input", input_id)


def check_overview(page, rows):
    expected = {(row.get("machine"), row["config"]["backend"], row["config"]["sink"]) for row in rows}
    assert page.locator(".study-plot").count() == len(expected)
    lines = page.locator(".frontier-line").evaluate_all("""nodes => nodes.map(node => ({
      condition: node.dataset.condition,
      input: node.dataset.input,
      rows: node.__data__.map(row => ({id: row.id, condition: row.condition, input: row.config.input_id}))
    }))""")
    frontier_ids = set(page.locator(".point.frontier").evaluate_all("nodes => nodes.map(node => node.dataset.id)"))
    assert {row["id"] for line in lines for row in line["rows"]} == frontier_ids
    for line in lines:
        assert {row["condition"] for row in line["rows"]} == {line["condition"]}
        assert {row["input"] for row in line["rows"]} == {line["input"]}
    colors = page.locator(".point").evaluate_all("""nodes => nodes.map(node => ({
      input: node.dataset.input, color: getComputedStyle(node.querySelector('.mark')).stroke
    }))""")
    by_input = {}
    for item in colors:
        by_input.setdefault(item["input"], set()).add(item["color"])
    assert all(len(values) == 1 for values in by_input.values())
    assert len({next(iter(values)) for values in by_input.values()}) == len(by_input)
    assert len(set(page.locator("#legend path").evaluate_all("nodes => nodes.map(node => node.getAttribute('d'))"))) == 5


def switch_without_jump(page, input_id):
    before = page.evaluate("({y: scrollY, time: performance.timeOrigin})")
    page.evaluate("window.previousPlots = [...document.querySelectorAll('.study-plot > svg')]")
    select_input(page, input_id)
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    after = page.evaluate("({y: scrollY, time: performance.timeOrigin})")
    assert abs(before["y"] - after["y"]) <= 1, (input_id, before, after)
    assert before["time"] == after["time"]
    assert page.evaluate("window.previousPlots.every((svg, i) => svg === document.querySelectorAll('.study-plot > svg')[i])")


def displayed_rows(datasets, report):
    if report is None:
        return [{**row, "machine": data["study"]["machine"]["name"]} for data in datasets for row in data["measurements"]]
    by_id = {data["study"]["id"]: data for data in datasets}
    return [{**row, "input_label": item["label"], "machine": by_id[source["study"]]["study"]["machine"]["name"]} for item in report for source in item["sources"]
            for row in by_id[source["study"]]["measurements"]
            if row["config"]["input_id"] == item["input"] and row["config"]["backend"] in source["backends"]]


def check_site(site, screenshots, executable=None):
    from playwright.sync_api import expect, sync_playwright

    index = json.loads((site / "data/microscopy/index.json").read_text())
    datasets = [json.loads((site / item["file"]).read_text()) for item in index["studies"]]
    rows = displayed_rows(datasets, index.get("report"))
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    screenshots.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            options = {"executable_path": str(executable)} if executable else {}
            browser = playwright.chromium.launch(headless=True, **options)
            context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
            page = context.new_page()
            errors, fetched = [], set()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: fetched.add(request.url))
            page.goto(f"{base}/microscopy.html?input=all&extent=all", wait_until="networkidle")
            if not rows:
                expect(page.locator("#load-status")).to_contain_text("No microscopy measurements")
                expect(page.locator("#workspace")).to_be_hidden()
                page.screenshot(path=str(screenshots / "empty.png"), full_page=True)
                browser.close()
                return
            expect(page.locator("#workspace")).to_be_visible()
            expect(page.locator("#table-body tr")).to_have_count(len(rows))
            assert page.locator(".point").count() == len(rows)
            assert page.locator("#study").count() == 0
            check_axes(page, rows)
            check_overview(page, rows)
            assert page.locator('.point[tabindex="-1"]').count() == 0
            previews = {(item["asset"], item["pack_sha256"]): item for item in index.get("previews", [])}
            for row in rows:
                preview = previews.get((row["config"]["image_asset_id"], row["detail"]["image_input"]["pack_sha256"]))
                if preview:
                    image = page.locator(f'#dataset-tabs button[data-input="{row["config"]["input_id"]}"] img').first
                    assert image.get_attribute("src") == preview["file"]
                    assert image.evaluate("image => image.complete && image.naturalWidth === 128")
            assert page.locator("#preview-sources figure").count() == len(previews)
            entropies = {(item["asset"], item["pack_sha256"]): item for item in index.get("entropy", {}).get("inputs", [])}
            if entropies:
                page.locator("#entropy-summary").click()
                for input_id in dict.fromkeys(row["config"]["input_id"] for row in rows):
                    row = next(row for row in rows if row["config"]["input_id"] == input_id)
                    sample = entropies.get((row["config"]["image_asset_id"], row["detail"]["image_input"]["pack_sha256"]))
                    cells = page.locator(f'#entropy-body tr[data-input="{input_id}"]')
                    expected = f"{sample['pixel_entropy_bits']:.2f}" if sample else "Not sampled"
                    assert cells.locator(".entropy-value").evaluate("cell => cell.firstChild.textContent") == expected
                    if sample:
                        expect(cells.locator(".entropy-unique")).to_have_text(f"{sample['unique_values']:,}")
                        expect(cells.locator(".entropy-pixels")).to_have_text(f"{sample['sample_pixels']:,}")
                        limited = sample["unique_values"] == sample["sample_pixels"] < math.prod(sample["shape"])
                        expect(cells.locator(".entropy-limit")).to_have_count(int(limited))
                page.screenshot(path=str(screenshots / "entropy-datasets.png"), full_page=True)
                page.locator("#entropy-summary").click()
            page.screenshot(path=str(screenshots / "all-datasets.png"), full_page=True)
            expect(page.locator("#axis-note")).to_contain_text("limits differ")
            expect(page.locator("#axis-note")).to_contain_text("fold below 1")
            assert {"blosc-lz4", "blosc-zstd", "lz4 (raw)", "zstd (raw)"} <= set(page.locator("#codec option").all_text_contents())
            if index.get("report"):
                source_ids = {source["study"] for item in index["report"] for source in item["sources"]}
                for item in index["studies"]:
                    assert (f"{base}/{item['file']}" in fetched) == (item["id"] in source_ids)
                visible = page.locator("body").inner_text().lower()
                assert "(v2)" not in visible and "confirmation" not in visible and "refinement" not in visible
            frontier_count = page.locator(".point.frontier").count()
            page.locator("#fit-frontier").click()
            assert page.locator(".point.frontier").count() == frontier_count
            assert page.locator('.point.frontier[tabindex="-1"]').count() == 0
            assert page.locator('.point.resampled-frontier[tabindex="-1"]').count() == 0
            page.reload(wait_until="networkidle")
            expect(page.locator("#fit-frontier")).to_have_attribute("aria-pressed", "true")
            for mode in ["all", "input"]:
                select_filter(page, "axes", mode)
                panels = page.locator(".study-plot").evaluate_all("""panels => panels.map(panel => ({
                  id: panel.querySelector('.point').dataset.id,
                  ticks: Array.from(panel.querySelectorAll('.plot-axis .tick text'), node => node.textContent)
                }))""")
                grouped = {}
                for panel in panels:
                    key = "all" if mode == "all" else by_id[panel["id"]]["config"]["input_id"]
                    if key in grouped:
                        assert panel["ticks"] == grouped[key]
                    grouped[key] = panel["ticks"]
            select_filter(page, "axes", "panel")
            for codec in ["blosc-lz4", "lz4", "blosc-zstd", "zstd"]:
                select_filter(page, "codec", codec)
                expected = [row for row in rows if row["config"]["codec"] == codec]
                expect(page.locator("#table-body tr")).to_have_count(len(expected))
                assert page.locator(".point").count() == len(expected)
            select_filter(page, "codec", "all")
            select_filter(page, "block", "16384")
            filtered = [row for row in rows if not row["config"]["codec"].startswith("blosc-")
                        or row["config"]["blosc_block_bytes"] == 16384]
            expect(page.locator("#table-body tr")).to_have_count(len(filtered))
            page.locator("#close-filters").click()
            page.locator("#measurements > summary").click()
            page.get_by_role("button", name="Raw LZ4", exact=True).first.click()
            expect(page.locator("#detail-content")).to_contain_text("Pipeline stages")
            expect(page.locator("#detail-content")).to_contain_text("compress")
            url = page.url
            page.reload(wait_until="networkidle")
            expect(page.locator("#table-body tr")).to_have_count(len(filtered))
            expect(page.locator("#detail-content")).to_contain_text("Raw LZ4")
            assert page.url == url
            page.locator("#measurements > summary").click()
            with page.expect_download() as download:
                page.locator("#download").click()
            exported = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text())))
            assert {row["id"] for row in exported} == {row["id"] for row in filtered}
            for item in exported:
                expected = by_id[item["id"]]
                assert item["study"] == expected["study_id"]
                assert int(item["observations"]) == expected["count"]
                assert float(item["logical_gibs_median"]) == expected["throughput"]["median"]
                assert float(item["logical_compression_fold"]) == expected["compression_fold"]
            select_filter(page, "codec", "zstd")
            expect(page.locator("#detail-content")).to_contain_text("outside the current filters")
            page.go_back(wait_until="networkidle")
            expect(page.locator("#table-body tr")).to_have_count(len(filtered))
            page.goto(f"{base}/microscopy.html", wait_until="networkidle")
            first_input = rows[0]["config"]["input_id"]
            expect(page.locator("#input")).to_have_value(first_input)
            expect(page.locator("#show-all")).to_have_attribute("aria-pressed", "true")
            assert page.locator('.point[tabindex="-1"]').count() == 0
            assert page.locator("#measurements").get_attribute("open") is None
            input_ids = list(dict.fromkeys(row["config"]["input_id"] for row in rows))
            page.set_viewport_size({"width": 1440, "height": 800})
            page.wait_for_timeout(200)
            page.evaluate("scrollTo(0, 180)")
            assert page.evaluate("scrollY") > 100
            for input_id in input_ids + ["all", first_input]:
                switch_without_jump(page, input_id)
                if input_id != "all" and entropies:
                    row = next(row for row in rows if row["config"]["input_id"] == input_id)
                    sample = entropies.get((row["config"]["image_asset_id"], row["detail"]["image_input"]["pack_sha256"]))
                    expected = f"{sample['pixel_entropy_bits']:.2f} bits/pixel" if sample else "not available"
                    if sample and sample["unique_values"] == sample["sample_pixels"] < math.prod(sample["shape"]):
                        expected += " · sample limited"
                    expect(page.locator("#entropy-summary")).to_have_text(f"Sampled pixel entropy: {expected}")
            before = page.evaluate("scrollY")
            page.go_back(wait_until="networkidle")
            expect(page.locator("#input")).to_have_value("all")
            assert abs(page.evaluate("scrollY") - before) <= 1
            page.go_forward(wait_until="networkidle")
            expect(page.locator("#input")).to_have_value(first_input)
            assert abs(page.evaluate("scrollY") - before) <= 1
            select_filter(page, "axes", "all")
            page.locator("#close-filters").click()
            ticks = page.locator(".study-plot").evaluate_all("panels => panels.map(panel => [...panel.querySelectorAll('.plot-axis .tick text')].map(node => node.textContent))")
            if len(input_ids) > 1:
                switch_without_jump(page, input_ids[1])
                assert page.locator(".study-plot").evaluate_all("panels => panels.map(panel => [...panel.querySelectorAll('.plot-axis .tick text')].map(node => node.textContent))") == ticks
            select_filter(page, "axes", "panel")
            page.locator("#close-filters").click()
            page.set_viewport_size({"width": 1440, "height": 1050})
            for input_id in input_ids:
                select_input(page, input_id)
                expected = [row for row in rows if row["config"]["input_id"] == input_id]
                expect(page.locator("#table-body tr")).to_have_count(len(expected))
                assert page.locator(".point").count() == len(expected)
                check_axes(page, expected)
                assert page.locator('.point.frontier[tabindex="-1"]').count() == 0
                assert page.locator(".study-plot").count() == len({row["condition"] for row in expected})
                page.locator(".point.frontier").first.focus()
                page.keyboard.press("Enter")
                expect(page.locator("#detail-content")).to_contain_text("Pipeline stages")
                page.evaluate("scrollTo(0, 0)")
                plots_box = page.locator("#plots").bounding_box()
                detail_box = page.locator("#detail").bounding_box()
                assert detail_box["x"] >= plots_box["x"] + plots_box["width"]
                assert abs(detail_box["y"] - plots_box["y"]) <= 1
                set_theme(page, "light")
                page.screenshot(path=str(screenshots / f"{input_id}.png"), full_page=True)
                page.get_by_text("Source and replay details", exact=True).click()
                link = page.get_by_role("link", name="Retained raw observations", exact=True)
                selected_id = page.locator(".point.selected").get_attribute("data-id")
                source = next(data for data in datasets if data["study"]["id"] == by_id[selected_id]["study_id"])
                assert link.get_attribute("href") == source["study"]["archive"]
            select_input(page, first_input)
            page.locator(".point.frontier").first.click()
            page.evaluate("scrollTo(0, 0)")
            set_theme(page, "dark")
            select_input(page, "all")
            check_overview(page, rows)
            page.screenshot(path=str(screenshots / "all-datasets-dark.png"), full_page=True)
            select_input(page, first_input)
            page.screenshot(path=str(screenshots / "desktop-dark.png"), full_page=True)
            set_theme(page, "light")
            page.screenshot(path=str(screenshots / "desktop-light.png"), full_page=True)
            for width in [1200, 768, 720, 390, 375, 320]:
                page.set_viewport_size({"width": width, "height": 900 if width > 700 else 844})
                page.wait_for_timeout(200)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                assert page.locator(".plot-axis text").evaluate_all("nodes => nodes.every(node => parseFloat(getComputedStyle(node).fontSize) >= 10)")
                if width <= 1100:
                    if page.locator("#close-detail").is_visible():
                        page.locator("#close-detail").click()
                page.evaluate("scrollTo(0, 0)")
                page.screenshot(path=str(screenshots / f"width-{width}.png"), full_page=True)
                assert page.locator(".study-plot").first.bounding_box()["y"] < (900 if width > 700 else 844), (width, page.locator(".study-plot").first.bounding_box())
                select_input(page, "all")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                page.screenshot(path=str(screenshots / f"overview-{width}.png"), full_page=True)
                assert page.locator(".study-plot").first.bounding_box()["y"] < (900 if width > 700 else 844), (width, page.locator(".study-plot").first.bounding_box())
                page.evaluate("scrollTo(0, 180)")
                switch_without_jump(page, first_input)
            touch = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True, reduced_motion="reduce")
            mobile = touch.new_page()
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.goto(f"{base}/microscopy.html", wait_until="networkidle")
            assert mobile.locator(".point circle").first.get_attribute("r") == "18"
            mobile.evaluate("scrollTo(0, 180)")
            if len(input_ids) > 1:
                switch_without_jump(mobile, input_ids[1])
            switch_without_jump(mobile, "all")
            check_overview(mobile, rows)
            switch_without_jump(mobile, first_input)
            mobile.locator(".point.frontier").first.tap()
            expect(mobile.locator("#close-detail")).to_be_visible()
            assert mobile.locator("#detail").bounding_box()["height"] <= 0.61 * 844
            mobile.screenshot(path=str(screenshots / "mobile-selection.png"), full_page=True)
            mobile.locator("#close-detail").tap()
            expect(mobile.locator("#detail")).to_be_hidden()
            select_filter(mobile, "backend", "gpu")
            mobile.locator("#close-filters").tap()
            expect(mobile.locator("#active-filters")).to_contain_text("GPU")
            mobile.reload(wait_until="networkidle")
            expect(mobile.locator("#backend")).to_have_value("gpu")
            if entropies:
                mobile.locator("#entropy-summary").tap()
                assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                mobile.screenshot(path=str(screenshots / "entropy-mobile.png"), full_page=True)
                mobile.locator("#entropy-summary").tap()
                cached = {**index, "entropy": {"version": 1, "inputs": [
                    {"asset": item["asset"], "pack_sha256": item["pack_sha256"], "byte_entropy_bits": [0.0]}
                    for item in entropies.values()]}}
                stale = context.new_page()
                stale.on("pageerror", lambda error: errors.append(str(error)))
                stale.route("**/data/microscopy/index.json", lambda route: route.fulfill(json=cached))
                stale.goto(f"{base}/microscopy.html?input=all&extent=all", wait_until="networkidle")
                expect(stale.locator("#workspace")).to_be_visible()
                expect(stale.locator("#dataset-entropy")).to_be_hidden()
                expect(stale.locator(".point")).to_have_count(len(rows))
                stale.close()
            for data in datasets:
                assert context.request.get(f"{base}/{data['study']['archive']}").ok
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
