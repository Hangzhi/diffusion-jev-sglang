"""Verify the public page and one real queued doodle prediction. No mocked answers."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from browser_sketch_demo import draw_cat
from playwright.async_api import async_playwright, expect


async def run(url, output):
    output.mkdir(parents=True, exist_ok=True)
    errors, responses, health = [], [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1080})
        page.on("pageerror", lambda error: errors.append(str(error)))

        async def observe(response):
            if "/health" in response.url:
                health.append(response.status)
            if "/api/jobs" in response.url:
                body = await response.json()
                if body.get("status") != "pending" or response.request.method == "POST":
                    responses.append({"status_code": response.status, "body": body})

        page.on("response", observe)
        await page.goto(url.rstrip("/") + "/#doodle")
        await expect(page.get_by_label("Draw a doodle", exact=True)).to_be_visible(timeout=60_000)
        await page.wait_for_timeout(6500)
        assert health == [200], health
        assert not responses, "Browsing the page must not submit a prediction"
        print("Page loads without starting a prediction; health polling stopped.", flush=True)
        await draw_cat(page)
        started = time.monotonic()
        await page.get_by_role("button", name="Guess doodle", exact=True).click()
        await expect(page.locator("button.evaluate")).to_have_attribute("aria-busy", "true")
        await expect(page.get_by_role("status")).to_contain_text("prediction", timeout=15_000)
        print("Submitted real drawing. Waiting for the GPU and result.", flush=True)
        await expect(page.locator("button.evaluate")).to_have_attribute(
            "aria-busy", "false", timeout=1_200_000
        )
        elapsed = time.monotonic() - started
        await page.wait_for_timeout(500)
        completed = [
            item["body"] for item in responses if item["body"].get("status") == "completed"
        ]
        assert completed, responses
        result = completed[-1]["result"]
        assert result["model"] == "google/diffusiongemma-26B-A4B-it"
        assert result["meta"]["probability_source"] == "self_conditioned_denoiser_logits"
        (output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
        await page.screenshot(path=str(output / "doodle-desktop.png"), full_page=True)
        await page.get_by_role("button", name="Try a sketch", exact=True).click()
        await expect(page.locator(".image-gallery img").first).to_be_visible(timeout=10_000)
        await page.wait_for_function(
            "[...document.querySelectorAll('.image-gallery img')].every(img => img.complete && img.naturalWidth > 0)"
        )
        images_ok = await page.locator(".image-gallery img").evaluate_all(
            "images => images.every(img => img.complete && img.naturalWidth > 0)"
        )
        assert images_ok
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.get_by_role("button", name="Flowers", exact=True).click()
        await expect(page.locator(".image-gallery img").first).to_be_visible(timeout=10_000)
        await page.locator(".image-gallery img").last.scroll_into_view_if_needed()
        await page.wait_for_function(
            "[...document.querySelectorAll('.image-gallery img')].every(img => img.complete && img.naturalWidth > 0)"
        )
        await page.screenshot(path=str(output / "flowers-mobile.png"), full_page=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        report = {
            "url": url,
            "elapsed_seconds": round(elapsed, 2),
            "real_prediction": result,
            "health_requests": len(health),
            "page_errors": errors,
            "gallery_images_loaded": images_ok,
            "mobile_overflow": False,
            "responses": responses,
        }
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"choice": result["answers"]["doodle"]["choice"], "elapsed_seconds": elapsed}
            ),
            flush=True,
        )
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path(".cache/cloud-demo"))
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output))
