"""Real browser acceptance: Gemma text, gallery classification, upload, and mobile layout."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/diffusiongemma"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1440, "height": 1120})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        await page.get_by_text("Ready", exact=True).wait_for(timeout=180000)
        assert await page.get_by_text("DiffusionGemma 26B A4B", exact=True).count() == 1
        responses = []

        async def evaluate():
            async with page.expect_response("**/v1/systemone", timeout=180000) as pending:
                await page.locator("button.evaluate").click()
            response = await pending.value
            assert response.status == 200, await response.text()
            body = await response.json()
            assert body["model"] == "google/diffusiongemma-26B-A4B-it"
            assert body["meta"]["probability_source"] == "self_conditioned_denoiser_logits"
            await page.get_by_text("Ready", exact=True).wait_for()
            responses.append(body)
            return body

        body = await evaluate()
        assert set(body["answers"]) == {"emoji", "positive", "intensity"}
        await page.screenshot(path=str(OUTPUT / "text-desktop.png"), full_page=True)
        await page.get_by_role("button", name="Image classification", exact=True).click()
        await page.get_by_role("button", name="Select flower image 1", exact=True).wait_for()
        assert await page.locator(".image-gallery button").count() == 12
        await page.get_by_role("button", name="Next", exact=True).click()
        await page.get_by_role("button", name="Select flower image 13", exact=True).wait_for()
        await page.get_by_role("button", name="Previous", exact=True).click()
        await page.get_by_label("Filter image category").select_option("rose")
        await page.wait_for_function(
            "document.querySelector('.gallery-footer').textContent.includes('95 test images')"
        )
        await page.get_by_role("button", name="Select flower image 1", exact=True).click()
        assert "Dataset label:" not in await page.locator(".selected-image").inner_text()
        body = await evaluate()
        assert "flower" in body["answers"]
        assert "Dataset label: rose" in await page.locator(".selected-image").inner_text()
        await page.screenshot(path=str(OUTPUT / "vision-desktop.png"), full_page=True)
        await page.get_by_role("button", name="JSON", exact=True).click()
        assert "candidate_mass" in await page.locator(".json-output").inner_text()
        manifest = json.loads((ROOT / "data/flowers/manifest.json").read_text())
        image = next(row for row in manifest["images"] if row["split"] == "dev")
        await page.get_by_label("Upload classification image").set_input_files(
            str(ROOT / "data/flowers/images" / (image["id"] + ".jpg"))
        )
        await page.wait_for_function(
            "document.querySelector('.selected-image img').src.startsWith('data:image/')"
        )
        assert "Dataset label:" not in await page.locator(".selected-image").inner_text()
        await evaluate()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path=str(OUTPUT / "vision-mobile.png"), full_page=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        (OUTPUT / "browser-smoke.json").write_text(
            json.dumps({"passed": True, "console_errors": errors, "responses": responses}, indent=2)
            + "\n"
        )
        await browser.close()
        print("Browser text, gallery, upload, and mobile checks passed")


asyncio.run(main())
