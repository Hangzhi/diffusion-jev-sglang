"""Exercise the live sketch demo and optionally record its real model response.

uvx --with playwright python scripts/browser_sketch_demo.py --record
Requires a running DiffusionGemma API, Playwright Chromium, and FFmpeg for recording.
"""

import argparse
import asyncio
import hashlib
import json
import math
import random
import tempfile
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from playwright.async_api import async_playwright, expect

# Deliberately uneven, hand-authored strokes. No perfect circles or mirrored features.
CAT_STROKES = [
    [(151, 186), (128, 150), (112, 98), (145, 115), (201, 158),
     (226, 147), (264, 143), (305, 160), (369, 95), (360, 190),
     (384, 218), (404, 266), (399, 295), (384, 323), (352, 349),
     (307, 363), (263, 367), (217, 359), (175, 343), (141, 312),
     (122, 277), (123, 243), (133, 209), (152, 188)],
    [(140, 158), (132, 121), (175, 168)],
    [(328, 177), (355, 126), (349, 183)],
    [(204, 221), (194, 223), (189, 234), (194, 243), (204, 245),
     (211, 236), (209, 226), (203, 222)],
    [(281, 232), (290, 222), (303, 230)],
    [(245, 257), (267, 259), (257, 274), (245, 257)],
    [(257, 274), (254, 290), (243, 299), (228, 292)],
    [(254, 289), (270, 302), (287, 287)],
    [(177, 260), (135, 246), (84, 250)],
    [(177, 282), (125, 285), (77, 301)],
    [(183, 301), (147, 321), (114, 346)],
    [(325, 261), (377, 239), (429, 235)],
    [(325, 282), (376, 286), (433, 302)],
    [(321, 302), (359, 322), (390, 347)],
]


async def draw_cat(page, *, paced=False, touch=False):
    canvas = page.get_by_label("Draw a doodle", exact=True)
    await canvas.scroll_into_view_if_needed()
    box = await canvas.bounding_box()
    assert box
    strokes = []
    rng = random.Random(23)
    for anchors in CAT_STROKES:
        stroke = [anchors[0]]
        for start, end in pairwise(anchors):
            steps = max(1, round(math.dist(start, end) / 8))
            for step in range(1, steps):
                t = step / steps
                stroke.append((start[0] + (end[0] - start[0]) * t + rng.uniform(-1.2, 1.2),
                               start[1] + (end[1] - start[1]) * t + rng.uniform(-1.2, 1.2)))
            stroke.append(end)
        strokes.append(stroke)
    session = await page.context.new_cdp_session(page) if touch else None
    button = page.locator("button.evaluate")
    for index, stroke in enumerate(strokes):
        points = [(box["x"] + x * box["width"] / 512,
                   box["y"] + y * box["height"] / 512) for x, y in stroke]
        if session:
            await session.send("Input.dispatchTouchEvent", {
                "type": "touchStart", "touchPoints": [{"x": points[0][0], "y": points[0][1]}]
            })
            await expect(button).to_be_disabled()
            if index:
                await expect(button).to_have_css("opacity", "1")
            for x, y in points[1:]:
                await session.send("Input.dispatchTouchEvent", {
                    "type": "touchMove", "touchPoints": [{"x": x, "y": y}]
                })
            await session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        else:
            await page.mouse.move(*points[0])
            await page.mouse.down()
            await expect(button).to_be_disabled()
            if index:
                await expect(button).to_have_css("opacity", "1")
            for x, y in points[1:]:
                await page.mouse.move(x, y)
                if paced:
                    await asyncio.sleep(0.015)
            await page.mouse.up()
        await expect(button).to_be_enabled()
        await expect(button).to_have_css("opacity", "1")
        if paced:
            await asyncio.sleep(0.14)
    if session:
        await session.detach()


async def evaluate(page, answer_key):
    async with page.expect_response("**/v1/systemone", timeout=180000) as pending:
        await page.locator("button.evaluate").click()
        await expect(page.locator("button.evaluate")).to_have_css("opacity", "1")
    response = await pending.value
    assert response.status == 200, await response.text()
    body = await response.json()
    assert body["model"] == "google/diffusiongemma-26B-A4B-it"
    assert body["meta"]["probability_source"] == "self_conditioned_denoiser_logits"
    assert answer_key in body["answers"]
    await expect(page.locator(".answer-value").first).to_be_visible()
    return body


async def run(args, videos):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    responses = {}
    checks = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        viewport = {"width": 1200, "height": 900}
        context = await browser.new_context(
            viewport=viewport,
            **({"record_video_dir": videos, "record_video_size": viewport} if args.record else {}),
        )
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.url, wait_until="networkidle")
        await expect(page.get_by_label("Draw a doodle", exact=True)).to_be_visible()
        assert page.url.endswith("#doodle")
        await expect(page.get_by_text("Ready", exact=True)).to_be_visible()
        await expect(page.locator("button.evaluate")).to_be_disabled()
        assert not await page.locator("details").evaluate("el => el.open")
        await page.evaluate("scrollTo(0, document.querySelector('.mode-tabs').offsetTop - 16)")
        if args.record:
            await asyncio.sleep(1)
        await draw_cat(page, paced=args.record)
        await expect(page.locator("button.evaluate")).to_be_enabled()
        # The request is built by the app from actual browser canvas pixels.
        responses["drawn_cat"] = await evaluate(page, "doodle")
        assert responses["drawn_cat"]["answers"]["doodle"]["choice"] == "cat"
        await page.screenshot(path=str(output / "sketch-desktop.png"))
        canvas_png = await page.locator("canvas").screenshot()
        (output / "drawn-cat.png").write_bytes(canvas_png)
        checks.append("Root opens drawing pad; mouse strokes produce a real cat prediction")
        checks.append("Button stays opaque between strokes; submitting is blocked until pointer-up")
        if args.record:
            await asyncio.sleep(4)
        video = page.video
        await context.close()
        if args.record:
            video_path = await video.path()
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                "-filter_complex",
                ("fps=12,scale=1000:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3"),
                "-loop", "0", str(output / "sketch-demo.gif"),
            )
            assert await process.wait() == 0, "FFmpeg conversion failed"

        context = await browser.new_context(viewport={"width": 1200, "height": 1000})
        page = await context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.url + "#doodle", wait_until="networkidle")
        blank = await page.locator("canvas").evaluate("el => el.toDataURL()")
        await draw_cat(page)
        await page.get_by_role("button", name="Doodle Detective", exact=True).click()
        await expect(page.locator("button.evaluate")).to_be_disabled()
        assert await page.locator("canvas").evaluate("el => el.toDataURL()") == blank
        await draw_cat(page)
        await page.get_by_role("button", name="Clear drawing", exact=True).click()
        await expect(page.locator("button.evaluate")).to_be_disabled()
        assert await page.locator("canvas").evaluate("el => el.toDataURL()") == blank
        assert await page.locator(".answer-value").count() == 0
        checks.append("Clear drawing and reopening the demo reset pixels and disable guessing")

        await page.get_by_role("button", name="Try a sketch", exact=True).click()
        await expect(page.locator(".image-gallery button")).to_have_count(12)
        await page.get_by_role("button", name="Next", exact=True).click()
        await expect(page.get_by_role("button", name="Select doodle 13", exact=True)).to_be_visible()
        await page.get_by_label("Filter image category").select_option("cat")
        await expect(page.locator(".gallery-footer")).to_contain_text("of 24 sketches")
        await page.get_by_role("button", name="Select doodle 1", exact=True).click()
        assert "Dataset label:" not in await page.locator(".selected-image").inner_text()
        responses["gallery_cat"] = await evaluate(page, "doodle")
        await expect(page.locator(".selected-image")).to_contain_text("Dataset label: cat")
        await page.locator("summary").click()
        await page.get_by_label("STATE", exact=True).fill("Identify the object in this sketch.")
        assert "Dataset label:" not in await page.locator(".selected-image").inner_text()
        checks.append("Gallery paging/filtering works; labels appear only after inference")

        await page.get_by_role("button", name="Flowers", exact=True).click()
        await page.get_by_label("Filter image category").select_option("rose")
        await page.get_by_role("button", name="Select flower image 1", exact=True).click()
        responses["flower"] = await evaluate(page, "flower")
        await page.reload(wait_until="networkidle")
        assert page.url.endswith("#flowers")
        await expect(page.get_by_role("button", name="Flowers", exact=True)).to_have_attribute(
            "aria-pressed", "true"
        )
        await page.get_by_role("button", name="Text decisions", exact=True).click()
        responses["text"] = await evaluate(page, "emoji")
        assert set(responses["text"]["answers"]) == {"emoji", "positive", "intensity"}
        await page.reload(wait_until="networkidle")
        await expect(page.get_by_role("button", name="Evaluate state", exact=True)).to_be_visible()
        checks.append("Flowers and text still evaluate; links preserve the selected demo on reload")

        # Missing optional data must never prevent drawing or uploaded-image inference.
        await page.route("**/api/datasets/quickdraw**", lambda route: route.fulfill(status=404))
        await page.get_by_role("button", name="Doodle Detective", exact=True).click()
        await expect(page.locator("canvas")).to_be_visible()
        await page.get_by_label("Upload classification image").set_input_files({
            "name": "drawing.png", "mimeType": "image/png", "buffer": canvas_png,
        })
        responses["upload_without_gallery"] = await evaluate(page, "doodle")
        await page.get_by_role("button", name="Draw your own", exact=True).click()
        await expect(page.locator("button.evaluate")).to_be_disabled()
        checks.append("Drawing and image upload work when gallery endpoints return 404")
        await context.close()

        # A slow examples request previously reset a demo chosen during startup.
        page = await browser.new_page()
        async def slow_examples(route):
            await asyncio.sleep(2)
            await route.continue_()
        await page.route("**/api/examples", slow_examples)
        await page.goto(args.url, wait_until="domcontentloaded")
        await page.get_by_role("button", name="Flowers", exact=True).click()
        await page.wait_for_load_state("networkidle")
        assert page.url.endswith("#flowers")
        checks.append("Delayed examples do not replace a demo the user already selected")
        await page.close()

        # Check the text-only backend navigation without starting a second GPU model.
        page = await browser.new_page()
        await page.route("**/health", lambda route: route.fulfill(json={
            "ready": True, "supports_images": False, "display_name": "Text-only UI check"
        }))
        await page.goto(args.url, wait_until="networkidle")
        await expect(page.get_by_role("button", name="Evaluate state", exact=True)).to_be_visible()
        await expect(page.get_by_role("button", name="Doodle Detective", exact=True)).to_be_disabled()
        checks.append("Simulated text-only health response retains the text-first interface")
        await page.close()

        mobile = await browser.new_context(
            viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
            device_scale_factor=1,
        )
        page = await mobile.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.url + "#doodle", wait_until="networkidle")
        await draw_cat(page, touch=True)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        responses["touch_cat"] = await evaluate(page, "doodle")
        assert responses["touch_cat"]["answers"]["doodle"]["choice"] == "cat"
        await page.screenshot(path=str(output / "sketch-mobile.png"), full_page=True)
        checks.append("390px touch drawing produces a real cat prediction, with no horizontal overflow")
        await mobile.close()
        assert not errors, errors
        browser_version = browser.version
        await browser.close()
    report = {
        "passed": True, "recorded_at": datetime.now(UTC).isoformat(),
        "browser": browser_version, "checks": checks, "console_errors": errors,
        "drawing_sha256": hashlib.sha256(canvas_png).hexdigest(), "responses": responses,
        "recording": "Real browser pointer input and unmodified GPU response; normal speed, 12 fps"
        if args.record else None,
    }
    (output / "sketch-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": checks, "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--output", type=Path, default=Path(".cache/sketch-demo"))
    parser.add_argument("--record", action="store_true")
    with tempfile.TemporaryDirectory() as videos:
        asyncio.run(run(parser.parse_args(), videos))
