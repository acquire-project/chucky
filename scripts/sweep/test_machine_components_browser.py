# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.62.0"]
# ///
"""Check registry-driven machine presentation across all four report pages."""
import argparse
import copy
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from site_server import ReportHandler


class QuietHandler(ReportHandler):
    def log_message(self, *args):
        pass


def check_site(site, screenshots, executable):
    from playwright.sync_api import expect, sync_playwright

    catalog = json.loads((site / "data/machines.json").read_text())
    experiments = json.loads((site / "data/pareto/index.json").read_text())["experiments"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=str(executable) if executable else None)
            context = browser.new_context(viewport={"width": 1440, "height": 1050})
            errors, pages = [], {}
            for name in ["index", "explore", "pareto", "microscopy"]:
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/{name}.html", wait_until="networkidle")
                expect(page.locator("machine-summary").first).to_be_visible()
                pages[name] = page
            for name in ["auk", "oreb", "reef-l40"]:
                pages["explore"].locator("#machine-sel").select_option(name)
                summaries = [page.locator(f'machine-summary[data-machine="{name}"]').first for page in pages.values()]
                for summary in summaries:
                    expect(summary).to_have_count(1)
                text = [summary.text_content() for summary in summaries]
                assert len(set(text)) == 1, (name, text)
            # Host selection groups runs, while individual selections retain URLs
            # and leave the other hosts' runs selected.
            page = pages["pareto"]
            expect(page.locator("#systems input")).to_have_count(len({experiment["machine_id"] for experiment in experiments}))
            page.locator("#settings > summary").click()
            page.locator('#runs input[value="blosc-rtx5070-20260905"]').uncheck()
            assert page.locator('#systems input[value="auk"]').evaluate("input => input.indeterminate")
            expect(page.locator('#systems input[value="oreb"]')).to_be_checked()
            page.reload(wait_until="networkidle")
            assert page.locator('#systems input[value="auk"]').evaluate("input => input.indeterminate")
            page.locator('#systems input[value="auk"]').check()
            expect(page.locator("#runs input:checked")).to_have_count(len(experiments))
            pages["explore"].locator("#machine-sel").select_option("oreb")
            for width in [1440, 390, 320]:
                for theme in ["light", "dark"]:
                    styles = []
                    for name, page in pages.items():
                        page.set_viewport_size({"width": width, "height": 1050})
                        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
                        summary = page.locator('machine-summary[data-machine="oreb"]').first
                        styles.append(summary.locator(".machine-spec").first.evaluate("node => {const s = getComputedStyle(node); return [s.fontFamily, s.fontSize, s.fontWeight, s.lineHeight, s.color];}"))
                        # Trend charts redraw after a debounced resize event.
                        page.wait_for_function("document.documentElement.scrollWidth <= innerWidth + 1", timeout=5000)
                        summary.screenshot(path=str(screenshots / f"{name}-{width}-{theme}.png"))
                        if width != 320 and theme == "light":
                            page.evaluate("scrollTo(0, 0)")
                            page.screenshot(path=str(screenshots / f"{name}-{width}-viewport.png"))
                    assert all(style == styles[0] for style in styles), styles
            # A catalog-only description change must reach every page, even
            # though all retained runs still contain their original GPU names.
            changed = copy.deepcopy(catalog)
            next(machine for machine in changed["machines"] if machine["name"] == "oreb")["specs"]["gpu"] = "Registry override GPU"
            for name, page in pages.items():
                page.route("**/data/machines.json", lambda route: route.fulfill(json=changed))
                page.reload(wait_until="networkidle")
                if name == "explore":
                    page.locator("#machine-sel").select_option("oreb")
                expect(page.locator('machine-summary[data-machine="oreb"]').first).to_contain_text("Registry override GPU")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print("Machine names, descriptions, styles and catalog updates agree across all four pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    check_site(args.site, args.screenshots, args.executable)
