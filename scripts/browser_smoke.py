"""Live browser acceptance test. Start the real API/engine first.

uvx --with playwright python scripts/browser_smoke.py
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(
            viewport={"width": 1440, "height": 1120}, device_scale_factor=1
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        await page.get_by_role("button", name="Evaluate state", exact=True).wait_for()
        await page.screenshot(path="reports/playground-empty.png", full_page=True)
        async with page.expect_response("**/v1/systemone", timeout=180000) as pending:
            await page.get_by_role("button", name="Evaluate state", exact=True).click()
        response = await pending.value
        assert response.status == 200, await response.text()
        body = await response.json()
        assert body["meta"]["probability_source"] == "masked_position_logits"
        assert set(body["answers"]) == {"emoji", "positive", "intensity"}
        await page.get_by_text("Real model logits", exact=False).wait_for()
        await page.get_by_text("Ready", exact=True).wait_for(timeout=30000)
        await page.screenshot(path="reports/playground-results.png", full_page=True)
        await page.get_by_role("button", name="JSON", exact=True).click()
        assert "candidate_logits" in await page.locator(".json-output").inner_text()
        await page.get_by_role("button", name="Support triage", exact=True).click()
        assert "Charged twice" in await page.locator("#context").input_value()
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(path="reports/playground-mobile.png", full_page=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        Path("reports/browser-smoke.json").write_text(
            json.dumps({"passed": True, "console_errors": errors, "response": body}, indent=2)
            + "\n"
        )
        await browser.close()


asyncio.run(main())
