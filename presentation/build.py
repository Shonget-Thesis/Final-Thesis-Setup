#!/usr/bin/env python3
"""Build the presentation walkthrough from modular slide files.

Usage:
    python presentation/build.py             # Build index.html once
    python presentation/build.py --watch     # Watch files and rebuild on save
    python presentation/build.py --serve     # Rebuild, watch, and start local dev server
    python presentation/build.py --sync-md   # Also update presentation/SCRIPT.md
"""

import argparse
import http.server
import re
import socketserver
import sys
import threading
import time
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent
SLIDES_DIR = PRESENTATION_DIR / "slides"
TEMPLATE_FILE = PRESENTATION_DIR / "template.html"
OUTPUT_FILE = PRESENTATION_DIR / "index.html"
SCRIPT_MD_FILE = PRESENTATION_DIR / "SCRIPT.md"


def parse_slide_file(path: Path):
    """Parse a modular slide HTML file into its components."""
    content = path.read_text(encoding="utf-8")

    # 1. Extract presenter script
    script_match = re.search(r'<script\s+type="text/slide-script">([\s\S]*?)</script>', content)
    script_text = script_match.group(1).strip() if script_match else ""

    # 2. Extract slide-specific CSS
    style_match = re.search(r'<style>([\s\S]*?)</style>', content)
    css_text = style_match.group(1).strip() if style_match else ""

    # 3. Extract extra JS (scripts not of type text/slide-script)
    js_blocks = []
    for m in re.finditer(r'<script(?!\s+type="text/slide-script")[^>]*>([\s\S]*?)</script>', content):
        js = m.group(1).strip()
        if js:
            js_blocks.append(js)
    extra_js = "\n\n".join(js_blocks)

    # 4. Extract slide HTML (<section class="scene ...> ... </section>)
    section_match = re.search(r'((?:<!--[^\n]*-->\s*)?<section class="scene[\s\S]*?</section>)', content)
    if not section_match:
        raise ValueError(f"Could not find <section class=\"scene\"> in {path.name}")
    html_section = section_match.group(1).strip()

    # Title & duration metadata from section tag
    title_match = re.search(r'data-title="([^"]*)"', html_section)
    dur_match = re.search(r'data-dur="([^"]*)"', html_section)
    ch_match = re.search(r'data-ch="([^"]*)"', html_section)

    meta = {
        "title": title_match.group(1) if title_match else path.stem,
        "duration": dur_match.group(1) if dur_match else "30",
        "chapter": ch_match.group(1) if ch_match else "0",
    }

    return {
        "html": html_section,
        "css": css_text,
        "script": script_text,
        "extra_js": extra_js,
        "meta": meta,
    }


def build_presentation(sync_md: bool = False) -> bool:
    """Compile template.html and slides/*.html into index.html."""
    t0 = time.perf_counter()

    if not TEMPLATE_FILE.exists():
        print(f"Error: Template file not found: {TEMPLATE_FILE}")
        return False

    slide_files = sorted(SLIDES_DIR.glob("*.html"))
    if not slide_files:
        print(f"Error: No slide files found in {SLIDES_DIR}")
        return False

    template = TEMPLATE_FILE.read_text(encoding="utf-8")

    all_html = []
    all_css = []
    all_scripts = []
    all_extra_js = []
    slide_data = []

    for path in slide_files:
        try:
            data = parse_slide_file(path)
            slide_data.append(data)
            all_html.append(data["html"])
            if data["css"]:
                all_css.append(data["css"])
            if data["extra_js"]:
                all_extra_js.append(data["extra_js"])
            # Format JS template literal, escaping backticks
            escaped_script = data["script"].replace("`", "\\`")
            all_scripts.append(f"      `{escaped_script}`")
        except Exception as exc:
            print(f"Error processing {path.name}: {exc}")
            return False

    # Insert components into template placeholders
    compiled = template

    # Replace CSS
    styles_str = "\n\n".join(all_css)
    compiled = compiled.replace("/* SLIDE_STYLES */", styles_str).replace("<!-- SLIDE_STYLES -->", styles_str)

    # Replace HTML
    slides_str = "\n\n        ".join(all_html)
    compiled = compiled.replace("<!-- SLIDES -->", slides_str)

    # Replace SCRIPT array items
    scripts_str = ",\n".join(all_scripts)
    compiled = compiled.replace("<!-- SLIDE_SCRIPTS -->", scripts_str)

    # Replace extra JS
    extra_js_str = "\n\n".join(all_extra_js)
    compiled = compiled.replace("/* SLIDE_EXTRA_JS */", extra_js_str)

    OUTPUT_FILE.write_text(compiled, encoding="utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Built {OUTPUT_FILE.name} with {len(slide_files)} slides in {elapsed_ms:.1f}ms")

    if sync_md:
        sync_script_md(slide_data)

    return True


def sync_script_md(slide_data: list):
    """Sync presentation/SCRIPT.md with the extracted slide scripts."""
    ch_names = {
        "0": "Proposal",
        "1": "Chapter I · Introduction",
        "2": "Chapter II · Review of related literature",
        "3": "Chapter III · Methodology",
    }
    md_lines = [
        "# Presenter script",
        "",
        "Read-aloud script for the animated walkthrough in [index.html](index.html). Timings are the auto-play durations of each scene (about 11 minutes in total).",
        "",
    ]
    for idx, s in enumerate(slide_data):
        meta = s["meta"]
        ch = ch_names.get(meta["chapter"], "Chapter " + meta["chapter"])
        dur = meta["duration"]
        md_lines.append(f"## Scene {idx+1:02d} · {meta['title']}")
        md_lines.append("")
        md_lines.append(f"*{ch} · about {dur} s*")
        md_lines.append("")
        md_lines.append(s["script"])
        md_lines.append("")

    SCRIPT_MD_FILE.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Synced {SCRIPT_MD_FILE.name}")


def watch_and_rebuild(sync_md: bool = False):
    """Watch slide files and template for changes and auto-rebuild."""
    print("Watching presentation/slides/ and presentation/template.html for changes... (Press Ctrl+C to stop)")
    mtimes = {}

    def get_tracked_files():
        files = [TEMPLATE_FILE]
        if SLIDES_DIR.exists():
            files.extend(SLIDES_DIR.glob("*.html"))
        return files

    for f in get_tracked_files():
        try:
            mtimes[f] = f.stat().st_mtime
        except OSError:
            pass

    while True:
        try:
            time.sleep(0.5)
            changed = False
            current_files = get_tracked_files()

            if set(current_files) != set(mtimes.keys()):
                changed = True

            for f in current_files:
                try:
                    m = f.stat().st_mtime
                    if mtimes.get(f) != m:
                        changed = True
                        mtimes[f] = m
                except OSError:
                    pass

            if changed:
                mtimes = {f: f.stat().st_mtime for f in current_files if f.exists()}
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Change detected, rebuilding...")
                build_presentation(sync_md=sync_md)

        except KeyboardInterrupt:
            print("\nStopped watcher.")
            break


def serve_presentation(port: int = 8000):
    """Run a local development HTTP server serving the presentation."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PRESENTATION_DIR), **kwargs)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving presentation at http://localhost:{port}/index.html")
        httpd.serve_forever()


def export_pdf(output_path: Path = None, port: int = 8765) -> bool:
    """Export slides to a vector PDF with selectable text using headless Edge/Chrome."""
    import asyncio
    import base64
    import json
    import shutil
    import subprocess
    import tempfile
    import urllib.request

    try:
        import websockets  # type: ignore
    except ImportError:
        print("Error: 'websockets' package is required for PDF export. Run: pip install websockets")
        return False

    target_pdf = output_path or (PRESENTATION_DIR / "proposal-defense-slides.pdf")

    # Locate Edge or Chrome
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    browser_exe = next((str(p) for p in candidates if p.exists()), None)
    if not browser_exe:
        print("Error: Could not find Microsoft Edge or Google Chrome executable.")
        return False

    print(f"Exporting PDF using {Path(browser_exe).name}...")

    # Start local HTTP server
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PRESENTATION_DIR), **kwargs)

        def log_message(self, format, *args):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", port), QuietHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    temp_profile = tempfile.mkdtemp(prefix="slide_pdf_")
    proc = None
    cdp_port = 9222

    async def _run_cdp():
        nonlocal proc
        cmd = [
            browser_exe,
            "--headless=new",
            f"--remote-debugging-port={cdp_port}",
            f"--user-data-dir={temp_profile}",
            "--disable-gpu",
            "--no-first-run",
            "about:blank",
        ]
        proc = subprocess.Popen(cmd)

        page_ws_url = None
        for _ in range(40):
            await asyncio.sleep(0.25)
            try:
                resp = urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/list", timeout=1)
                targets = json.loads(resp.read().decode())
                for t in targets:
                    if t.get("type") == "page":
                        page_ws_url = t.get("webSocketDebuggerUrl")
                        break
                if page_ws_url:
                    break
            except Exception:
                pass

        if not page_ws_url:
            raise RuntimeError("Failed to obtain Page WebSocket URL from browser.")

        async with websockets.connect(page_ws_url, max_size=100 * 1024 * 1024) as ws:
            msg_id = 1

            async def call(method, params=None):
                nonlocal msg_id
                cid = msg_id
                msg_id += 1
                payload = {"id": cid, "method": method}
                if params:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("id") == cid:
                        if "error" in data:
                            raise RuntimeError(f"CDP error: {data['error']}")
                        return data.get("result", {})

            await call("Page.enable")
            await call("Runtime.enable")
            await call("Page.navigate", {"url": f"http://127.0.0.1:{port}/index.html"})

            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("method") == "Page.loadEventFired":
                    break

            js_prep = """
            (async () => {
                await document.fonts.ready;
                if (window.prepareForPrint) {
                    window.prepareForPrint();
                }
                return document.querySelectorAll('.scene').length;
            })()
            """
            res = await call("Runtime.evaluate", {"expression": js_prep, "awaitPromise": True})
            slide_count = res.get("result", {}).get("value", 0)

            await asyncio.sleep(2.0)

            pdf_res = await call("Page.printToPDF", {
                "landscape": True,
                "displayHeaderFooter": False,
                "printBackground": True,
                "preferCSSPageSize": True,
                "paperWidth": 16.666667,  # 1600px / 96dpi
                "paperHeight": 9.375,     # 900px / 96dpi
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
                "generateTaggedPDF": True,
            })

            data_b64 = pdf_res["data"]
            pdf_bytes = base64.b64decode(data_b64)
            target_pdf.write_bytes(pdf_bytes)
            print(f"Successfully exported {slide_count} slides to {target_pdf.name} ({len(pdf_bytes):,} bytes)")

    try:
        asyncio.run(_run_cdp())
        return True
    except Exception as exc:
        print(f"PDF export error: {exc}")
        return False
    finally:
        if proc:
            proc.terminate()
            proc.wait()
        httpd.shutdown()
        shutil.rmtree(temp_profile, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build modular presentation slides into index.html")
    parser.add_argument("--watch", action="store_true", help="Watch for file changes and auto-rebuild")
    parser.add_argument("--serve", action="store_true", help="Start local development server and watch")
    parser.add_argument("--port", type=int, default=8000, help="Port for dev server (default: 8000)")
    parser.add_argument("--sync-md", action="store_true", help="Also sync presentation/SCRIPT.md")
    parser.add_argument("--pdf", action="store_true", help="Export presentation slides to PDF with selectable text")

    args = parser.parse_args()

    # Initial build
    success = build_presentation(sync_md=args.sync_md)
    if not success:
        sys.exit(1)

    if args.pdf:
        pdf_ok = export_pdf()
        if not pdf_ok:
            sys.exit(1)

    if args.serve:
        server_thread = threading.Thread(target=serve_presentation, args=(args.port,), daemon=True)
        server_thread.start()
        watch_and_rebuild(sync_md=args.sync_md)
    elif args.watch:
        watch_and_rebuild(sync_md=args.sync_md)


if __name__ == "__main__":
    main()

