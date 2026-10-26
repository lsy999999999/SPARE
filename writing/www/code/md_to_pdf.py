#!/usr/bin/env python3
"""Small dependency-free Markdown-to-HTML wrapper for LibreOffice PDF export."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def inline(text: str, base: Path) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^]]*)\]\(([^)]+)\)",
                  lambda m: f'<img src="{(base / html.unescape(m.group(2))).resolve()}" alt="{m.group(1)}">', text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def is_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def render(md: str, source: Path) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    paragraph: list[str] = []
    def flush() -> None:
        if paragraph:
            out.append("<p>" + " ".join(inline(x.strip(), source.parent) for x in paragraph) + "</p>")
            paragraph.clear()
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush(); i += 1; continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush(); level = len(heading.group(1)); out.append(f"<h{level}>{inline(heading.group(2), source.parent)}</h{level}>"); i += 1; continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and "|" in lines[i + 1] and is_separator(lines[i + 1]):
            flush()
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            out.append('<table><thead><tr>' + ''.join(f'<th>{inline(c, source.parent)}</th>' for c in headers) + '</tr></thead><tbody>')
            for row in rows:
                out.append('<tr>' + ''.join(f'<td>{inline(c, source.parent)}</td>' for c in row) + '</tr>')
            out.append('</tbody></table>')
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            flush(); items = []
            while i < len(lines):
                m = re.match(r"^\s*[-*]\s+(.*)$", lines[i])
                if not m: break
                items.append(f"<li>{inline(m.group(1), source.parent)}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>"); continue
        if re.match(r"^```", line):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1; out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>"); continue
        paragraph.append(line); i += 1
    flush()
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: md_to_pdf.py INPUT.md OUTPUT.html")
    source, target = map(Path, sys.argv[1:])
    body = render(source.read_text(encoding="utf-8"), source)
    title = source.stem
    target.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
@page {{ size: A4; margin: 18mm 15mm 18mm 15mm; }}
body {{ font-family: "Noto Sans CJK SC", "Noto Serif CJK SC", sans-serif; color: #202124; font-size: 10pt; line-height: 1.5; }}
h1 {{ font-size: 20pt; border-bottom: 1.5px solid #555; padding-bottom: 5pt; }}
h2 {{ font-size: 14pt; margin-top: 18pt; border-bottom: 0.7px solid #aaa; padding-bottom: 2pt; }}
h3 {{ font-size: 11.5pt; }}
p {{ margin: 6pt 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 8pt 0 12pt; font-size: 7.2pt; page-break-inside: auto; }}
tr {{ page-break-inside: avoid; }} th, td {{ border: 0.5px solid #999; padding: 3pt 4pt; vertical-align: top; }}
th {{ background: #e9eef5; font-weight: bold; }}
img {{ max-width: 100%; height: auto; display: block; margin: 8pt auto; page-break-inside: avoid; }}
code {{ font-family: "Noto Sans Mono CJK SC", monospace; font-size: 8.5pt; background: #f1f3f4; padding: 1pt 2pt; }}
pre {{ background: #f1f3f4; padding: 6pt; white-space: pre-wrap; font-size: 8pt; }}
a {{ color: #164d8f; }} ul {{ margin-top: 4pt; }}
</style></head><body>{body}</body></html>''', encoding="utf-8")


if __name__ == "__main__":
    main()
