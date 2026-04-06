#!/usr/bin/env python3
"""
Generate Spillover_Alpha_Project_Summary.pdf from README + DY explainer.
Uses markdown → HTML → PDF via weasyprint.
"""

import markdown
from weasyprint import HTML
from pathlib import Path
import base64
import re

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "Spillover_Alpha_Project_Summary.pdf"


def image_to_data_uri(match):
    """Convert markdown image references to base64 data URIs for embedding."""
    alt = match.group(1)
    src = match.group(2)
    img_path = ROOT / src
    if not img_path.exists():
        return match.group(0)
    suffix = img_path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix.lstrip("."), "image/png")
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return f"![{alt}](data:{mime};base64,{b64})"


def build_markdown():
    """Combine README + DY explainer into one markdown document."""
    readme = (ROOT / "README.md").read_text()
    dy = (ROOT / "docs" / "dy_framework_explained.md").read_text()

    # Replace the DY link in Chapter 5 with a reference to the appendix
    readme = readme.replace(
        "> Full technical walkthrough: **[docs/dy_framework_explained.md](docs/dy_framework_explained.md)**",
        "> Full technical walkthrough: see **Appendix A** below."
    )

    # Embed all images as base64
    img_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    readme = re.sub(img_pattern, image_to_data_uri, readme)
    dy = re.sub(img_pattern, image_to_data_uri, dy)

    # Combine
    combined = readme + "\n\n---\n\n"
    combined += "## Appendix A — The Diebold-Yilmaz Spillover Network: A Technical Walkthrough\n\n"
    # Strip the H1 title from DY doc (we replaced it with our own heading)
    dy_lines = dy.split("\n")
    if dy_lines[0].startswith("# "):
        dy_lines = dy_lines[1:]
    # Also strip the first ">" subtitle line if present
    while dy_lines and (dy_lines[0].strip() == "" or dy_lines[0].startswith(">")):
        dy_lines.pop(0)
    combined += "\n".join(dy_lines)

    return combined


CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {
        content: "Spillover Alpha — Team 54 | Page " counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 9px;
        color: #71717A;
    }
}
body {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
}
h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #111;
    border-bottom: 3px solid #A855F7;
    padding-bottom: 8px;
    margin-top: 0;
}
h2 {
    font-size: 16pt;
    font-weight: 700;
    color: #1a1a1a;
    border-bottom: 1px solid #e5e5e5;
    padding-bottom: 4px;
    margin-top: 28px;
    page-break-after: avoid;
}
h3 {
    font-size: 13pt;
    font-weight: 600;
    color: #333;
    margin-top: 20px;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    font-weight: 600;
    color: #444;
    margin-top: 14px;
    page-break-after: avoid;
}
p { margin: 6px 0; }
blockquote {
    border-left: 3px solid #A855F7;
    padding: 8px 14px;
    margin: 12px 0;
    background: #faf5ff;
    color: #444;
    font-size: 10pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 10pt;
}
th {
    background: #f3f0ff;
    border: 1px solid #d4d4d8;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
}
td {
    border: 1px solid #e5e5e5;
    padding: 5px 10px;
}
tr:nth-child(even) { background: #fafafa; }
code {
    font-family: 'IBM Plex Mono', 'Fira Code', monospace;
    font-size: 9pt;
    background: #f4f4f5;
    padding: 1px 4px;
    border-radius: 3px;
}
pre {
    background: #18181B;
    color: #e5e5e5;
    padding: 14px;
    border-radius: 6px;
    font-size: 9pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: none;
    padding: 0;
    color: #e5e5e5;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12px auto;
    border-radius: 4px;
    border: 1px solid #e5e5e5;
}
hr {
    border: none;
    border-top: 1px solid #e5e5e5;
    margin: 24px 0;
}
strong { color: #111; }
a { color: #7C3AED; text-decoration: none; }
"""


def main():
    print("Building combined markdown...")
    md_text = build_markdown()

    print("Converting to HTML...")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    full_html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>{CSS}</style>
</head><body>
{html_body}
</body></html>"""

    print(f"Rendering PDF → {OUTPUT.name} ...")
    HTML(string=full_html, base_url=str(ROOT)).write_pdf(str(OUTPUT))
    print(f"Done! {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
