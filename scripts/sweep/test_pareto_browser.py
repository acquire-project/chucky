# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.62.0"]
# ///
"""Exercise a built site in Chromium; save desktop/mobile screenshots in both themes.

Build report.py first. Then: uv run scripts/sweep/test_pareto_browser.py
Uses installed Edge on Windows, Playwright Chromium elsewhere.
"""
import argparse
import csv
import io
import json
import re
import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from site_server import ReportHandler


class QuietHandler(ReportHandler):
    def log_message(self, *args):
        pass


def check_site(site, screenshots, executable=None):
    from playwright.sync_api import sync_playwright, expect
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            options = {"executable_path": str(executable)} if executable else {"channel": "msedge" if sys.platform == "win32" else None}
            browser = p.chromium.launch(headless=True, **options)
            index = json.loads((site / "data/pareto/index.json").read_text())
            system_count = len(index["experiments"])
            machine_count = len({experiment["machine_id"] for experiment in index["experiments"]})
            measurement_count = sum(experiment["configuration_count"] for experiment in index["experiments"])
            context = browser.new_context(viewport={"width": 1560, "height": 1050}, accept_downloads=True)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: print(f"Browser {m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: print(f"Browser exception: {e}"))
            page.goto(base + "/pareto.html")
            expect(page.locator("#workspace")).to_be_visible()
            expect(page.locator("#systems input:checked")).to_have_count(1)
            expect(page.locator("#systems input:checked")).to_have_value("reef-l40")
            expect(page.locator("#layout")).to_have_value("overlay")
            expect(page.locator("#mode")).to_have_value("cross")
            expect(page.locator(".plot-cell")).to_have_count(1)
            expect(page.locator(".pareto-intro")).to_contain_text("Pareto frontier analysis")
            default_count = page.locator("#table-body tr").count()
            page.goto(base + "/pareto.html?systems=all&layout=matrix&workload=all&mode=codec")
            expect(page.locator("#workspace")).to_be_visible()
            expect(page.locator("#scenario-note")).to_contain_text("orca2_single")
            expect(page.locator("#scenario-note a")).to_have_attribute("href", re.compile(r"bench_stream_orca2_single\.c$"))
            expect(page.locator("#systems input:checked")).to_have_count(machine_count)
            expect(page.locator("#systems")).to_contain_text("CPU")
            expect(page.locator("#systems")).to_contain_text("Storage")
            assert not re.search(r"\d{4}-\d{2}-\d{2}", page.locator("#systems").inner_text())
            page.get_by_role("button", name="Run date (UTC)", exact=True).click()
            page.reload()
            expect(page.locator('#table-head th[aria-sort="descending"]')).to_contain_text("Run date (UTC)")
            page.get_by_role("button", name="Input GiB/s", exact=True).click()
            expect(page.locator(".plot-cell")).to_have_count(4 * system_count)
            expect(page.locator("#table-body tr")).to_have_count(measurement_count)
            expect(page.locator("#raw")).to_have_count(0)
            expect(page.locator("#failure-note")).to_contain_text("1 configuration")
            page.get_by_role("button", name="Inspect a failed configuration").click()
            expect(page.locator("#detail-content")).to_contain_text("excluded from frontier membership")
            expect(page.locator("#detail-content")).to_contain_text("Warmup failed: out-of-memory")
            expect(page.locator("#table-body tr.selected")).to_contain_text("Warmup failed")
            assert page.locator(".chart-point.selected").count() == 1
            assert page.locator("#table-body tr.selected").get_attribute("data-id").startswith("blosc-auk-")
            with page.expect_download() as failure_download:
                page.locator("#download").click()
            exported = list(csv.DictReader(io.StringIO(Path(failure_download.value.path()).read_text())))
            failed = next(row for row in exported if row["status"] == "partial")
            assert failed["machine"] == "auk" and failed["frontier"] == "false"
            assert failed["run_date_utc"] == "2026-09-16"
            assert json.loads(failed["failures_json"])[0]["kind"] == "out-of-memory"
            if page.locator("#settings").get_attribute("open") is None:
                page.locator("#settings > summary").click()
            for checkbox in page.locator("#runs input").all():
                checkbox.set_checked(checkbox.input_value() == "blosc-auk-20260916")
            expect(page.locator("#table-body tr")).to_have_count(200)
            expect(page.locator(".plot-cell")).to_have_count(4)
            page.reload()
            expect(page.locator("#runs input:checked")).to_have_count(1)
            assert page.locator('#systems input[value="auk"]').evaluate("input => input.indeterminate")
            for checkbox in page.locator("#systems input").all():
                checkbox.check()
            assert page.evaluate("d3.version") == "7.9.0"
            expect(page.locator("#scale")).to_have_count(0)
            expect(page.locator("#chart-note")).to_contain_text("logarithmic")
            # A logarithmic compression axis never begins at zero. Random-data
            # marks should use the available width instead of clustering at an edge.
            for chart in page.locator(".plot-cell > svg").all():
                assert "0" not in [s.strip() for s in chart.locator(".plot-axis").first.locator("text").all_text_contents()]
            random_span = page.locator(".workload-row").filter(has_text="12-bit random").first.locator(".plot-cell > svg").first.evaluate("""svg => {
                const xs = [...svg.querySelectorAll('.chart-point')].map(n => +n.getAttribute('transform').match(/translate\\(([^,]+)/)[1]);
                return (Math.max(...xs) - Math.min(...xs)) / svg.viewBox.baseVal.width;
            }""")
            assert random_span > .55, f"Random-data plot uses only {random_span:.0%} of chart width"
            for row in page.locator(".matrix-row").all():
                axes = [cell.locator(".plot-axis").all_text_contents() for cell in row.locator(".plot-cell").all()]
                assert all(axis == axes[0] for axis in axes), "System axes must align within each row"
            # Keyboard selection, linked table and persistent details.
            chart = page.locator(".plot-cell > svg").first
            chart.focus()
            chart.press("ArrowRight")
            expect(page.locator("#table-body tr.selected")).to_have_count(1)
            expect(page.locator(".chart-point.selected")).to_have_count(1)
            expect(page.locator("#detail-content")).to_contain_text("Min–max")
            reported = page.locator("#detail-content dd").all_text_contents()[:7]
            for value in reported:
                for token in re.findall(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, re.I):
                    mantissa = token.lower().split("e", 1)[0].replace(".", "").lstrip("0")
                    assert len(mantissa) <= 3, f"Detail value exceeds three significant figures: {value}"
            expect(page.locator("#detail-content summary")).to_have_text("Record references")
            assert "source_metrics" not in page.locator("#detail-content").inner_text()
            page.locator("#table-body tr.selected").get_attribute("data-id")
            assert "selected=" in page.url
            chart.press("End")
            chart.press("Home")
            # Focus moves within the table by arrows, with one tab stop for rows.
            page.locator('.setting-button[tabindex="0"]').focus()
            page.keyboard.press("ArrowDown")
            expect(page.locator('.setting-button[tabindex="0"]')).to_be_focused()
            # Filter, sort and CSV contents agree with the visible table.
            if page.locator("#settings").get_attribute("open") is None:
                page.locator("#settings > summary").click()
            page.locator("#codecs").select_option(["lz4"])
            page.locator("#shuffles").select_option(["bit"])
            page.locator("#budget").fill("2.5")
            page.locator("#budget").press("Tab")
            expect(page.locator("#table-body tr").first).to_be_visible()
            filtered_count = page.locator("#table-body tr").count()
            assert 0 < filtered_count < 576
            page.get_by_role("button", name="Estimated GiB", exact=False).click()
            with page.expect_download() as download:
                page.locator("#download").click()
            records = list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding="utf-8"))))
            assert len(records) == filtered_count
            assert all(r["codec"] == "blosc-lz4" and float(r["estimated_device_gib"]) <= 2.5 for r in records)
            assert all("summary.csv" in r["summary"] for r in records)
            # Selection survives a filter that hides it and a URL round trip.
            current_url = page.url
            page.reload()
            expect(page.locator("#workspace")).to_be_visible()
            assert page.url == current_url
            expect(page.locator("#table-body tr")).to_have_count(filtered_count)
            page.locator("#budget").fill("0")
            page.locator("#budget").press("Tab")
            expect(page.locator("#empty")).to_be_visible()
            expect(page.locator("#download")).to_be_disabled()
            expect(page.locator("#detail-content")).to_contain_text("outside the current filters")
            page.go_back()
            expect(page.locator("#table-body tr")).to_have_count(filtered_count)
            page.locator("#reset-filters").click()
            # The memory view directly shows estimated device allocations.
            expect(page.locator("#view")).to_have_value("compression")
            expect(page.locator("#memory")).to_have_count(0)
            page.locator("#view").select_option("memory")
            expect(page.locator("#view")).to_have_value("memory")
            expect(page.locator(".axis-title").first).to_have_text("Estimated device allocation (GiB)")
            page.locator("#view").select_option("compression")
            expect(page.locator("#table-body tr")).to_have_count(default_count)
            page.locator("#reset-filters").click()
            page.locator("#layout").select_option("overlay")
            expect(page.locator(".plot-cell")).to_have_count(1)
            expect(page.locator('#mode option[value="memory"]')).to_have_count(0)
            page.locator("#mode").select_option("cross")
            page.locator("#view").select_option("memory")
            expect(page.locator(".axis-title").first).to_have_text("Estimated device allocation (GiB)")
            page.locator("#fit").click()
            expect(page.locator("#chart-note")).to_contain_text("Zoomed")
            page.locator("#full").click()
            expect(page.locator("#chart-note")).to_contain_text("Full extent")
            expect(page.locator("#chart-note")).to_contain_text("logarithmic")
            page.screenshot(path=str(screenshots / "memory-overlay.png"), full_page=True)
            # Desktop/mobile in both themes, real touch selection on mobile.
            for mobile in (False, True):
                screen = {"width": 390, "height": 844} if mobile else {"width": 1560, "height": 1050}
                review = browser.new_context(viewport=screen, is_mobile=mobile, has_touch=mobile, device_scale_factor=1)
                tab = review.new_page()
                tab.on("pageerror", lambda e: errors.append(str(e)))
                tab.goto(base + "/pareto.html")
                expect(tab.locator("#workspace")).to_be_visible()
                for theme in ("light", "dark"):
                    if tab.locator("html").get_attribute("data-theme") != theme:
                        tab.locator("#theme-toggle").click()
                    assert tab.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Page clips horizontally"
                    tab.screenshot(path=str(screenshots / f"{'mobile' if mobile else 'desktop'}-{theme}.png"), full_page=True)
                    tab.screenshot(path=str(screenshots / f"{'mobile' if mobile else 'desktop'}-{theme}-viewport.png"))
                if mobile:
                    tab.locator(".plot-cell").first.locator(".chart-point .mark").last.tap()
                    expect(tab.locator("#detail-content")).to_contain_text("Median input throughput")
                # The actual SVG text must stay within its viewport, at every size.
                assert tab.evaluate("""() => [...document.querySelectorAll('.plot-cell > svg')].every(svg => {
                    const box = svg.getBoundingClientRect();
                    return [...svg.querySelectorAll('text')].every(t => {
                      const r = t.getBoundingClientRect();
                      return r.left >= box.left - 1 && r.right <= box.right + 1 && r.top >= box.top - 1 && r.bottom <= box.bottom + 1;
                    });
                })"""), "Chart labels clip"
                review.close()
            # Navigation and local D3 work on the existing pages too.
            for href, label in [("index.html", "Over time"), ("explore.html", "Benchmark explorer"), ("pareto.html", "Blosc GPU Analysis")]:
                page.locator(f'.site-head nav a[href="{href}"]').click()
                expect(page.locator('.site-head nav a[aria-current="page"]')).to_have_text(label)
                page.wait_for_load_state("networkidle")
                assert page.evaluate("d3.version") == "7.9.0"
            # Missing index is actionable, and retry succeeds after repair.
            page.route("**/data/pareto/index.json", lambda route: route.fulfill(status=503, body="unavailable"))
            page.reload()
            expect(page.locator("#load-status")).to_contain_text("Unable to load")
            expect(page.locator("#retry")).to_be_visible()
            page.unroute("**/data/pareto/index.json")
            page.locator("#retry").click()
            expect(page.locator("#workspace")).to_be_visible()
            # An empty manifest is a valid empty state.
            page.route("**/data/pareto/index.json", lambda route: route.fulfill(json={"version":1,"definitions":{},"experiments":[]}))
            page.reload()
            expect(page.locator("#empty")).to_contain_text("No retained datasets")
            page.unroute("**/data/pareto/index.json")
            # Schema-compatible future experiment: no hardcoded system count or label.
            idx = json.loads((site / "data/pareto/index.json").read_text())
            fourth = json.loads((site / idx["experiments"][0]["data"]).read_text())
            fourth["experiment"].update(id="fourth", label="Additional system", machine_id="additional-host", data="data/pareto/fourth.json")
            for r in fourth["measurements"]:
                r["experiment_id"] = "fourth"
                r["id"] = "fourth:" + r["configuration_id"]
            idx["experiments"].append(fourth["experiment"])
            page.route("**/data/pareto/index.json", lambda route: route.fulfill(json=idx))
            page.route("**/data/pareto/fourth.json", lambda route: route.fulfill(json=fourth))
            page.goto(base + "/pareto.html?systems=all&layout=matrix&workload=all")
            expect(page.locator("#systems input:checked")).to_have_count(machine_count + 1)
            expect(page.locator(".plot-cell")).to_have_count(4 * (system_count + 1))
            assert not errors, errors
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print(f"Browser checks passed; screenshots: {screenshots}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("_site"))
    parser.add_argument("--screenshots", type=Path, default=Path(".cache/pareto-browser-review"))
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    check_site(args.site, args.screenshots, args.executable)
