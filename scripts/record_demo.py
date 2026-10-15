"""Record the demo flow against the live site into results/demo.gif (no Chrome extension needed).

  .venv/bin/python scripts/record_demo.py https://dabba-jgho.onrender.com
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "demo_frames"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
W, H = 1180, 820
frames: list[tuple[Path, int, str]] = []  # (path, duration_ms, caption)


def shot(page, name: str, hold_ms: int, caption: str):
    p = OUT / f"{len(frames):03d}_{name}.png"
    page.screenshot(path=str(p))
    frames.append((p, hold_ms, caption))


def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle", timeout=120000)
        page.wait_for_selector("#sample", timeout=60000)
        page.wait_for_timeout(1000)
        shot(page, "landing", 1800, "Dabba: paste tonight's WhatsApp messages")
        page.click("#sample"); page.wait_for_timeout(600)
        shot(page, "sample", 2200, "12 messages from the kitchen's group, Hinglish and all")
        page.click("#ingest")
        t0 = time.time()
        while time.time() - t0 < 120:
            page.wait_for_timeout(1500)
            shot(page, "parsing", 700, "Qwen3.5-4B + Dabba LoRA reads each one (served by Tinker)")
            if page.locator("#rows tr").count() >= 10:
                break
        page.wait_for_timeout(800)
        shot(page, "parsed", 2600, "Every message is now a structured order with the right date")
        page.locator("#rows").scroll_into_view_if_needed(); page.wait_for_timeout(400)
        shot(page, "parsed2", 1800, "Harsh Khan said '1 dabba'. No rule yet, so it defaulted to veg thali")
        # wait for the plan
        t0 = time.time()
        while time.time() - t0 < 120 and "TabPFN" not in (page.locator("#prepsum").inner_text() or ""):
            page.wait_for_timeout(1500)
        page.locator("#prepsum").scroll_into_view_if_needed(); page.wait_for_timeout(500)
        shot(page, "plan", 3200, "TabPFN: tomorrow's count per dish with an 80% band, and what to cook")
        page.mouse.wheel(0, 320); page.wait_for_timeout(500)
        shot(page, "shopping", 2400, "...and a shopping list in rotis and kilos")
        # correct Harsh Khan's dish -> memory
        page.locator("#rows").scroll_into_view_if_needed(); page.wait_for_timeout(400)
        row = page.locator("#rows tr", has_text="Harsh Khan").first
        sel = row.locator("select.dish").first
        sel.select_option("paneer_thali")
        page.wait_for_timeout(900)
        shot(page, "corrected", 2600, "Change it once: 'Yaad rakh liya', Dabba remembers Harsh takes paneer")
        page.wait_for_timeout(2500)
        page.locator("#rules").scroll_into_view_if_needed(); page.wait_for_timeout(500)
        shot(page, "memory", 3000, "The rule lives in Backboard memory, readable and deletable")
        b.close()

    # assemble GIF with captions
    font = None
    for cand in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            font = ImageFont.truetype(cand, 26); break
        except Exception:
            pass
    imgs, durs = [], []
    for p, d, cap in frames:
        im = Image.open(p).convert("RGB")
        dr = ImageDraw.Draw(im, "RGBA")
        tw = dr.textlength(cap, font=font) if font else len(cap) * 12
        x0, y0 = 20, H - 64
        dr.rectangle([x0, y0, x0 + tw + 28, y0 + 46], fill=(30, 27, 23, 225))
        dr.text((x0 + 14, y0 + 9), cap, fill=(255, 255, 255), font=font)
        imgs.append(im.resize((W * 3 // 4, H * 3 // 4), Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durs.append(d)
    out = ROOT / "results" / "demo.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print("frames:", len(imgs), "->", out, f"{out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
