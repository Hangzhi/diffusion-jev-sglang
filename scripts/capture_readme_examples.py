"""Capture flower and emoji examples from the running local app and real model."""

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright, expect


async def capture(url, output):
    output.mkdir(parents=True, exist_ok=True)
    examples, errors = [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 1200})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url.rstrip("/") + "/#flowers")
        await expect(page.get_by_text("Ready", exact=True)).to_be_visible(timeout=60_000)
        async with page.expect_response(
            lambda response: "/flowers/images?" in response.url and "label=daisy" in response.url
        ):
            await page.get_by_label("Filter image category").select_option("daisy")
        await page.get_by_role("button", name="Select flower image 1", exact=True).click()

        for name in ("flower", "emoji"):
            if name == "emoji":
                await page.get_by_role("button", name="Text decisions", exact=True).click()
                await page.get_by_role("button", name="Emoji classifier", exact=True).click()
            async with page.expect_response("**/v1/systemone", timeout=120_000) as pending:
                await page.locator("button.evaluate").click()
            response = await pending.value
            assert response.ok, await response.text()
            result = await response.json()
            assert result["model"] == "google/diffusiongemma-26B-A4B-it"
            assert result["meta"]["probability_source"] == "self_conditioned_denoiser_logits"
            await expect(page.locator("button.evaluate")).to_have_attribute("aria-busy", "false")
            await expect(page.locator(".results")).to_be_visible()
            await page.evaluate("document.fonts.ready")
            if name == "flower":
                await page.locator(".image-gallery img").last.scroll_into_view_if_needed()
                await page.wait_for_function(
                    "[...document.querySelectorAll('.vision-panel img')]"
                    ".every(img => img.complete && img.naturalWidth > 0)"
                )
            await page.locator(".workspace").screenshot(path=str(output / f"{name}.png"))
            request = response.request.post_data_json
            if request.get("images"):
                # Record the gallery ID, never embed uploaded pixels in this report.
                assert all(not image.startswith("data:") for image in request["images"])
            examples.append({"task": name, "request": request, "response": result})
            print(f"Captured {name}: {result['answers'][name]['choice']}", flush=True)
        assert not errors, errors
        (output / "examples.json").write_text(
            json.dumps({"url": url, "examples": examples, "page_errors": errors}, indent=2)
            + "\n"
        )
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("reports/readme-demo"))
    args = parser.parse_args()
    asyncio.run(capture(args.url, args.output))
