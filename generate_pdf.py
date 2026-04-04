#!/usr/bin/env python3
"""Convert README.md to a styled PDF with embedded images."""

import markdown
from weasyprint import HTML
from pathlib import Path
import base64
import re

ROOT = Path(__file__).parent
README = ROOT / "README.md"
OUTPUT = ROOT / "Spillover_Alpha_Project_Summary.pdf"


def embed_images(html: str) -> str:
    """Replace local image paths with base64 data URIs so they render in PDF."""
    def replace_src(match):
        src = match.group(1)
        img_path = ROOT / src
        if img_path.exists():
            suffix = img_path.suffix.lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:{mime};base64,{b64}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', replace_src, html)


def main():
    md_text = README.read_text(encoding="utf-8")

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    # Embed local images as base64
    html_body = embed_images(html_body)

    # Wrap in styled HTML document
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 2cm 2.2cm;
        @bottom-center {{
            content: counter(page);
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 9px;
            color: #999;
        }}
    }}

    body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
        max-width: 100%;
    }}

    h1 {{
        font-size: 26pt;
        font-weight: 700;
        color: #0a0a0a;
        border-bottom: 3px solid #A855F7;
        padding-bottom: 8px;
        margin-top: 1.5em;
    }}

    h2 {{
        font-size: 18pt;
        font-weight: 600;
        color: #1a1a1a;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 6px;
        margin-top: 1.8em;
        page-break-after: avoid;
    }}

    h3 {{
        font-size: 14pt;
        font-weight: 600;
        color: #333;
        margin-top: 1.4em;
        page-break-after: avoid;
    }}

    h4 {{
        font-size: 12pt;
        font-weight: 600;
        color: #555;
        margin-top: 1.2em;
        page-break-after: avoid;
    }}

    p {{
        margin: 0.6em 0;
    }}

    blockquote {{
        border-left: 4px solid #A855F7;
        padding: 8px 16px;
        margin: 1em 0;
        background: #faf5ff;
        color: #444;
        font-style: italic;
    }}

    code {{
        font-family: 'SF Mono', 'Menlo', 'Courier New', monospace;
        font-size: 9.5pt;
        background: #f4f4f5;
        padding: 1px 4px;
        border-radius: 3px;
        color: #7c3aed;
    }}

    pre {{
        background: #18181b;
        color: #e4e4e7;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 9pt;
        line-height: 1.5;
        overflow-x: auto;
        page-break-inside: avoid;
    }}

    pre code {{
        background: none;
        padding: 0;
        color: inherit;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1em 0;
        font-size: 10pt;
        page-break-inside: avoid;
    }}

    th {{
        background: #f4f4f5;
        font-weight: 600;
        text-align: left;
        padding: 8px 10px;
        border: 1px solid #e5e5e5;
    }}

    td {{
        padding: 6px 10px;
        border: 1px solid #e5e5e5;
    }}

    tr:nth-child(even) {{
        background: #fafafa;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 1em auto;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
    }}

    hr {{
        border: none;
        border-top: 1px solid #e5e5e5;
        margin: 2em 0;
    }}

    strong {{
        color: #0a0a0a;
    }}

    /* Title page styling */
    h1:first-of-type {{
        font-size: 32pt;
        text-align: center;
        border-bottom: none;
        margin-top: 3cm;
        margin-bottom: 0.3em;
    }}

    /* First blockquote as subtitle */
    h1:first-of-type + blockquote {{
        text-align: center;
        border-left: none;
        background: none;
        font-style: normal;
        font-size: 12pt;
        color: #666;
    }}
</style>
</head>
<body>
<div style="text-align:center;margin-bottom:0.5cm;">
    <span style="font-family:monospace;font-size:9pt;color:#A855F7;letter-spacing:2px;">
        SPILLOVER ALPHA — PROJECT SUMMARY
    </span>
</div>
{html_body}
</body>
</html>"""

    HTML(string=html_doc, base_url=str(ROOT)).write_pdf(str(OUTPUT))
    print(f"Saved: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
