"""Revise Figure 1's legend, direction arrows, and KV titles."""

from pathlib import Path
import argparse
import base64
import copy
import io
import re
import xml.etree.ElementTree as ET

import cairosvg
import fitz
from PIL import ImageFont


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fixed_kv_23_paper_style_with_s3_arrowstyle_inward_arialchain.svg"
OUTPUT = HERE / "figure1_legend_large.svg"
PDF = HERE / "figure1_legend_large.pdf"
ORIGINAL_PDF = (HERE.parent / "www/manuscript/mainresults"
                / "figure1_for_mainresults_cropped.pdf")
TARGET = (HERE.parent / "www/latex_latest/www(5)/figures/main_results"
          / "figure1_for_mainresults_cropped.pdf")
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
FONT_SIZE = 17.5
MARKER_TO_TEXT = 13.5
ITEM_GAP = 11.0
ROW_GAP_PT = 6.0
BOTTOM_EXTRA_PT = 10.0 + 3 * ROW_GAP_PT
SVG_SCALE = 944.61 / 1612.097994
MARKER_Y = 820.283312 + BOTTOM_EXTRA_PT / SVG_SCALE
ARROW_COLOR = (0.20, 0.22, 0.25)
ARROW_SHIFT_PT = 3.0


def extract_arial_fonts(document: fitz.Document) -> tuple[bytes, bytes]:
    fonts = {base_font: xref for xref, _, _, base_font, *_ in document[0].get_fonts(full=True)}
    try:
        regular_xref = next(xref for name, xref in fonts.items() if "ArialMT" in name)
        bold_xref = next(xref for name, xref in fonts.items() if "Arial-BoldMT" in name)
    except StopIteration as exc:
        raise ValueError("The source PDF does not contain embedded Arial Regular and Bold") from exc
    regular = document.extract_font(regular_xref)[3]
    bold = document.extract_font(bold_xref)[3]
    if not regular or not bold:
        raise ValueError("Could not extract the embedded Arial fonts")
    return regular, bold


def embed_and_normalize_svg_fonts(root: ET.Element, regular_font: bytes) -> None:
    defs = root.find(f"{{{SVG}}}defs")
    if defs is None:
        defs = ET.Element(f"{{{SVG}}}defs")
        root.insert(0, defs)
    css = ET.SubElement(defs, f"{{{SVG}}}style", {"type": "text/css"})
    regular_b64 = base64.b64encode(regular_font).decode("ascii")
    css.text = (
        "@font-face {font-family: 'Arial'; font-style: normal; font-weight: 400; "
        f"src: url(data:font/ttf;base64,{regular_b64}) format('truetype');}}"
    )
    for element in root.iter():
        style = element.get("style")
        if style and "font-family" in style:
            element.set(
                "style",
                re.sub(r"font-family:\s*[^;]+", "font-family: 'Arial'", style),
            )


def add_svg_arrows_and_lower_titles(root: ET.Element) -> None:
    axes = [group for group in root.iter(f"{{{SVG}}}g")
            if group.get("id", "").startswith("axes_")]
    if len(axes) != 23:
        raise ValueError(f"Expected 23 panels, found {len(axes)}")
    for index, panel in enumerate(axes, 1):
        patches = [group for group in panel
                   if group.get("id", "").startswith("patch_")]
        titles = [group.find(f"{{{SVG}}}text") for group in panel
                  if group.get("id", "").startswith("text_")
                  and group.find(f"{{{SVG}}}text") is not None
                  and (group.find(f"{{{SVG}}}text").text or "").startswith("KV = ")]
        old_label = [group for group in panel
                     if group.find(f"{{{SVG}}}text") is not None
                     and group.find(f"{{{SVG}}}text").text == "Better"]
        if len(patches) < 3 or len(titles) != 1 or len(old_label) != 1:
            raise ValueError(f"Unexpected panel structure: {index}")
        bounds = [float(value) for value in patches[0].find(f"{{{SVG}}}path")
                  .get("d").replace("M", " ").replace("L", " ")
                  .replace("z", " ").split()]
        x0 = min(bounds[::2])
        y0 = min(bounds[1::2])
        for group in (*patches[-2:], *old_label):
            panel.remove(group)
        title = titles[0]
        x = (min(bounds[::2]) + max(bounds[::2])) / 2
        y = max(bounds[1::2]) + 26
        title.set("x", f"{x:.6f}")
        title.set("y", f"{y:.6f}")
        title.set("transform", f"rotate(-0 {x:.6f} {y:.6f})")

        arrow = ET.SubElement(panel, f"{{{SVG}}}g", {"id": f"direction_arrow_{index}"})
        ax = x0 + 27.3
        arrow_shift = ARROW_SHIFT_PT / SVG_SCALE
        ET.SubElement(arrow, f"{{{SVG}}}path", {
            "d": (f"M {ax:.6f} {y0 + 66 - arrow_shift:.6f} "
                  f"L {ax:.6f} {y0 + 35 - arrow_shift:.6f}"),
            "style": "fill: none; stroke: #333840; stroke-width: 2.7; "
                     "stroke-linecap: round",
        })
        ET.SubElement(arrow, f"{{{SVG}}}path", {
            "d": (f"M {ax - 6:.6f} {y0 + 38 - arrow_shift:.6f} "
                  f"L {ax:.6f} {y0 + 29 - arrow_shift:.6f} "
                  f"L {ax + 6:.6f} {y0 + 38 - arrow_shift:.6f} z"),
            "style": "fill: #333840",
        })
        ET.SubElement(arrow, f"{{{SVG}}}text", {
            "x": f"{ax + 17:.6f}",
            "y": f"{y0 + 47 - arrow_shift:.6f}",
            "style": "font-size: 15px; font-family: 'Arial'; fill: #333840",
        }).text = "better"
        row = (index - 1) // 6
        panel.set("transform", f"translate(0 {row * ROW_GAP_PT / SVG_SCALE:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true",
                        help="update the PDF referenced by the manuscript")
    args = parser.parse_args()
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XLINK)
    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
    ET.register_namespace("cc", "http://creativecommons.org/ns#")
    ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    figure_pdf = fitz.open(ORIGINAL_PDF)
    arial_regular, arial_bold = extract_arial_fonts(figure_pdf)
    embed_and_normalize_svg_fonts(root, arial_regular)
    add_svg_arrows_and_lower_titles(root)
    legend = root.find(f".//{{{SVG}}}g[@id='legend_1']")
    if legend is None:
        raise ValueError("Legend not found")
    markers = [group.find(f".//{{{SVG}}}use") for group in legend
               if group.get("id", "").startswith("line2d_")]
    texts = [group.find(f"{{{SVG}}}text") for group in legend
             if group.get("id", "").startswith("text_")]
    if len(markers) != 12 or len(texts) != 12 or any(x is None for x in markers + texts):
        raise ValueError("Expected 12 marker and text pairs")

    font = ImageFont.truetype(io.BytesIO(arial_regular), round(FONT_SIZE * 10))
    widths = [font.getlength(text.text or "") / 10 for text in texts]
    total = sum(MARKER_TO_TEXT + width for width in widths) + ITEM_GAP * 11
    cursor = (float(root.get("viewBox").split()[2]) - total) / 2
    for marker, label, width in zip(markers, texts, widths):
        new_x = cursor
        marker.set("x", f"{new_x:.6f}")
        marker.set("transform", f"translate({new_x:.6f} {MARKER_Y:.6f}) "
                   f"scale(1.35) translate({-new_x:.6f} {-MARKER_Y:.6f})")
        marker.set("y", f"{MARKER_Y:.6f}")
        label_x = new_x + MARKER_TO_TEXT
        label_y = MARKER_Y + FONT_SIZE * .35
        label.set("x", f"{label_x:.6f}")
        label.set("y", f"{label_y:.6f}")
        label.set("transform", f"rotate(-0 {label_x:.6f} {label_y:.6f})")
        label.set("style", label.get("style", "").replace(
            "font-size: 9.5px", f"font-size: {FONT_SIZE}px"
        ) + "; font-weight: 700; stroke: #000000; stroke-width: 0.35px; "
            "paint-order: stroke fill")
        cursor += MARKER_TO_TEXT + width + ITEM_GAP

    root.set("width", "944.61pt")
    root.set("height", f"{508.22 + BOTTOM_EXTRA_PT}pt")
    view = list(map(float, root.get("viewBox").split()))
    view[3] += BOTTOM_EXTRA_PT / SVG_SCALE
    root.set("viewBox", " ".join(map(str, view)))
    background = root.find(f".//{{{SVG}}}g[@id='patch_1']/{{{SVG}}}path")
    background.set("d", f"M 0 {view[3]} L {view[2]} {view[3]} L {view[2]} 0 L 0 0 z")
    xlabel = next(t for t in root.iter(f"{{{SVG}}}text") if t.text == "Mean TTFT (ms)")
    xlabel.set("y", str(float(xlabel.get("y")) + BOTTOM_EXTRA_PT / SVG_SCALE))
    xlabel.attrib.pop("transform", None)
    ylabel = next(t for t in root.iter(f"{{{SVG}}}text")
                  if t.text == "Mean Serving Capacity (QPS)")
    ylabel.set("transform", f"translate(0 {1.5 * ROW_GAP_PT / SVG_SCALE:.6f}) "
               + ylabel.get("transform", ""))
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)
    page = figure_pdf[0]
    original_spans = [span for block in page.get_text("dict")["blocks"]
                      if "lines" in block for line in block["lines"]
                      for span in line["spans"]]
    titles = sorted((span for span in original_spans
                     if span["text"].startswith("KV = ")),
                    key=lambda span: (span["bbox"][1], span["bbox"][0]))
    old_arrows = sorted((span for span in original_spans
                         if span["text"] == "Better"),
                        key=lambda span: (span["bbox"][1], span["bbox"][0]))
    xlabel_span = next(span for span in original_spans if span["text"] == "Mean TTFT (ms)")
    ylabel_span = next(span for span in original_spans
                       if span["text"] == "Mean Serving Capacity (QPS)")
    panels = [drawing["rect"] for drawing in page.get_drawings()
              if drawing["type"] == "f" and drawing["fill"] == (1, 1, 1)
              and 140 < drawing["rect"].width < 145
              and 95 < drawing["rect"].height < 100]
    if len(titles) != 23 or len(old_arrows) != 23 or len(panels) != 23:
        raise ValueError("Expected 23 PDF titles, arrows, and plot panels")
    grid = [drawing for drawing in page.get_drawings()
            if drawing["type"] == "f" and drawing["fill"] is not None
            and all(abs(channel - .69) < .005 for channel in drawing["fill"])
            and (drawing["rect"].width > 20 or drawing["rect"].height > 20)]
    arrow_boxes = []
    for title, old_arrow in zip(titles, old_arrows):
        page.add_redact_annot(fitz.Rect(title["bbox"]), fill=None)
        x0, y0, x1, y1 = old_arrow["bbox"]
        rect = fitz.Rect(x0 - 8.7, y0 - 11.5, x1 + 8, y1 + 1)
        arrow_boxes.append(rect)
        page.add_redact_annot(rect, fill=(1, 1, 1))
    band = fitz.Rect(0, 468, page.rect.width, 488)
    legend_band = band + (0, BOTTOM_EXTRA_PT, 0, BOTTOM_EXTRA_PT)
    scale = page.rect.width / float(root.get("viewBox").split()[2])
    legend_only = ET.Element(f"{{{SVG}}}svg", {
        "width": f"{band.width}pt",
        "height": f"{band.height}pt",
        "viewBox": f"0 {legend_band.y0 / scale:.6f} {band.width / scale:.6f} "
                   f"{band.height / scale:.6f}",
        "version": "1.1",
    })
    marker_only_legend = copy.deepcopy(legend)
    for child in list(marker_only_legend):
        if child.get("id", "").startswith("text_"):
            marker_only_legend.remove(child)
    legend_only.append(marker_only_legend)
    legend_pdf = fitz.open(
        stream=cairosvg.svg2pdf(bytestring=ET.tostring(legend_only)), filetype="pdf"
    )
    page.add_redact_annot(fitz.Rect(0, 472, page.rect.width, 488), fill=None)
    page.add_redact_annot(fitz.Rect(xlabel_span["bbox"]), fill=None)
    page.add_redact_annot(fitz.Rect(ylabel_span["bbox"]), fill=None)
    page.apply_redactions(images=0, graphics=1, text=0)

    for rect in arrow_boxes:
        for drawing in grid:
            intersection = rect & drawing["rect"]
            if not intersection.is_empty:
                page.draw_rect(intersection, color=None, fill=drawing["fill"],
                               fill_opacity=drawing["fill_opacity"])
    page.insert_font(fontname="arial_regular", fontbuffer=arial_regular)
    for panel in panels:
        x = panel.x0 + 16
        y = panel.y0 - ARROW_SHIFT_PT
        page.draw_line((x, y + 39), (x, y + 21),
                       color=ARROW_COLOR, width=1.55)
        page.draw_polyline([(x - 3.5, y + 22), (x, y + 17),
                            (x + 3.5, y + 22)],
                           color=ARROW_COLOR, fill=ARROW_COLOR, closePath=True)
        page.insert_text((x + 10, y + 28), "better", fontname="arial_regular",
                         fontsize=8.8, color=ARROW_COLOR)
    final_pdf = fitz.open()
    final_page = final_pdf.new_page(width=page.rect.width,
                                    height=page.rect.height + BOTTOM_EXTRA_PT)
    # Leave room after each row's KV labels. Strip text from other rows before
    # importing each vector band so PDF text extraction contains each label once.
    row_starts = [panels[i].y0 for i in (0, 6, 12, 18)]
    cuts = [0] + [top - 5 for top in row_starts[1:]] + [page.rect.height]
    row_documents = []
    for row in range(4):
        row_doc = fitz.open()
        row_doc.insert_pdf(figure_pdf)
        row_page = row_doc[0]
        if cuts[row] > 0:
            row_page.add_redact_annot(fitz.Rect(0, 0, page.rect.width, cuts[row]), fill=None)
        if cuts[row + 1] < page.rect.height:
            row_page.add_redact_annot(fitz.Rect(0, cuts[row + 1], page.rect.width,
                                               page.rect.height), fill=None)
        row_page.apply_redactions(images=0, graphics=2, text=0)
        clip = fitz.Rect(0, cuts[row], page.rect.width, cuts[row + 1])
        destination = clip + (0, row * ROW_GAP_PT, 0, row * ROW_GAP_PT)
        final_page.show_pdf_page(destination, row_doc, 0, clip=clip)
        row_documents.append(row_doc)
    final_page.insert_font(fontname="arial_bold", fontbuffer=arial_bold)
    title_font = fitz.Font(fontbuffer=arial_bold)
    for index, (title, panel) in enumerate(zip(titles, panels)):
        title_width = title_font.text_length(title["text"], fontsize=8.3)
        final_page.insert_text(((panel.x0 + panel.x1 - title_width) / 2,
                                panel.y1 + 16.5 + (index // 6) * ROW_GAP_PT),
                               title["text"], fontname="arial_bold", fontsize=8.3)
    final_page.show_pdf_page(legend_band, legend_pdf, 0)
    final_page.insert_font(fontname="arial_legend", fontbuffer=arial_regular)
    for label in texts:
        final_page.insert_text(
            (float(label.get("x")) * scale, float(label.get("y")) * scale),
            label.text or "",
            fontname="arial_legend",
            fontsize=FONT_SIZE * scale,
            color=(0, 0, 0),
            fill=(0, 0, 0),
            render_mode=2,
            border_width=0.05,
        )
    final_page.insert_font(fontname="arial_axis", fontbuffer=arial_regular)
    xlabel_width = fitz.Font(fontbuffer=arial_regular).text_length(
        xlabel_span["text"], fontsize=xlabel_span["size"])
    final_page.insert_text(((page.rect.width - xlabel_width) / 2,
                            xlabel_span["origin"][1] + BOTTOM_EXTRA_PT),
                           xlabel_span["text"], fontname="arial_axis", fontsize=xlabel_span["size"])
    final_page.insert_text((ylabel_span["origin"][0],
                            ylabel_span["origin"][1] + 1.5 * ROW_GAP_PT),
                           ylabel_span["text"], fontname="arial_axis",
                           fontsize=ylabel_span["size"], rotate=90)
    final_pdf.save(PDF, garbage=4, deflate=True)
    final_pdf.close()
    for row_doc in row_documents:
        row_doc.close()
    figure_pdf.close()
    legend_pdf.close()
    if args.install:
        TARGET.write_bytes(PDF.read_bytes())
        print(TARGET)
    print(f"Legend SVG font size: 9.5 -> {FONT_SIZE} px bold; "
          f"span {total:.1f} / 1612.1")
    print(OUTPUT)
    print(PDF)


if __name__ == "__main__":
    main()
