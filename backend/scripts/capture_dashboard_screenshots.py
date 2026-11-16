"""
Capture live full-page PNG screenshots of all 7 pages of the EPI-BMA Dashboard
using Playwright headless Chromium against http://127.0.0.1:5173.
"""
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = ROOT_DIR / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("1. Overview", "screenshot_page1_overview.png"),
    ("2. Data Explorer", "screenshot_page2_data_explorer.png"),
    ("3. Model Analysis", "screenshot_page3_model_analysis.png"),
    ("4. Outbreak Forecast", "screenshot_page4_outbreak_forecast.png"),
    ("5. Uncertainty Analysis", "screenshot_page5_uncertainty_analysis.png"),
    ("6. Model Evaluation", "screenshot_page6_model_evaluation.png"),
    ("7. About the Model", "screenshot_page7_about_model.png"),
]


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        page.goto("http://127.0.0.1:5173", wait_until="networkidle", timeout=30000)
        time.sleep(1.5)

        for tab_label, filename in PAGES:
            btn = page.get_by_role("button", name=tab_label)
            btn.click()
            page.wait_for_load_state("networkidle")
            time.sleep(1.2)
            out_path = ASSETS_DIR / filename
            page.screenshot(path=str(out_path), full_page=True)
            print(f"Captured {tab_label} -> {out_path}")

        browser.close()


if __name__ == "__main__":
    capture_all()
