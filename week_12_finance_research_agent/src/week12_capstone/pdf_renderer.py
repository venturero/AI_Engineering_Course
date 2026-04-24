from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BCG_GREEN = colors.HexColor("#147B58")
TEXT_COLOR = colors.HexColor("#1E1E1E")
SUBTLE_GRAY = colors.HexColor("#5C6670")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            name="Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=TEXT_COLOR,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            name="Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=SUBTLE_GRAY,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            name="Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=BCG_GREEN,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            name="Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=TEXT_COLOR,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            name="Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=TEXT_COLOR,
            leftIndent=14,
            bulletIndent=3,
            spaceAfter=4,
        ),
        "subsection": ParagraphStyle(
            name="Subsection",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEXT_COLOR,
            spaceBefore=6,
            spaceAfter=3,
        ),
    }


def _iter_sections(markdown_text: str) -> Iterable[tuple[str, list[str]]]:
    current_title = None
    current_lines = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current_title:
                yield current_title, current_lines
            current_title = line.replace("## ", "", 1).strip()
            current_lines = []
        elif current_title:
            current_lines.append(raw_line.rstrip())
    if current_title:
        yield current_title, current_lines


def _normalize_ticker(raw_ticker: str) -> str:
    cleaned = (raw_ticker or "").strip()
    if not cleaned:
        return ""
    primary = cleaned.split(",", 1)[0].strip()
    if ":" in primary:
        primary = primary.split(":", 1)[1]
    primary = primary.strip().upper()
    primary = primary.replace("UNKNOWN", "").strip()
    if not primary:
        return ""
    if primary.startswith("$"):
        return primary
    return f"${primary}"


def _company_to_ticker(stock_calls: Optional[List[Dict[str, object]]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for call in stock_calls or []:
        company = str(call.get("company", "")).strip()
        if not company:
            continue
        ticker = _normalize_ticker(str(call.get("ticker", "")))
        if ticker:
            mapping[company] = ticker
    return mapping


def _inject_tickers(text: str, ticker_map: Dict[str, str]) -> str:
    if not text or not ticker_map:
        return text
    updated = text
    for company, ticker in sorted(ticker_map.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"\b{re.escape(company)}\b(?!\s*\()", re.IGNORECASE)
        updated = pattern.sub(lambda m: f"{m.group(0)} ({ticker})", updated)
    return updated


def _markdown_to_reportlab_html(text: str) -> str:
    escaped = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SUBTLE_GRAY)
    w, _h = doc.pagesize
    canvas.drawString(doc.leftMargin, 10 * mm, "Confidential — for discussion purposes only")
    canvas.drawRightString(w - doc.rightMargin, 10 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def render_strategy_pdf(
    report_markdown: str,
    user_query: str,
    output_path: Path,
    executive_takeaways: Optional[List[str]] = None,
    stock_calls: Optional[List[Dict[str, object]]] = None,
    chart_path: Optional[Path] = None,
    visual_path: Optional[Path] = None,
    visual_error: Optional[str] = None,
    visual_caption: Optional[str] = None,
) -> None:
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ticker_map = _company_to_ticker(stock_calls)
    report_markdown = _inject_tickers(report_markdown, ticker_map)
    enriched_takeaways = [_inject_tickers(item, ticker_map) for item in (executive_takeaways or [])]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        title="Strategy perspectives",
        author="Week 12 Capstone",
    )

    story = []
    cover_image_ok = bool(visual_path and visual_path.exists())

    if cover_image_ok:
        story.append(Image(str(visual_path), width=160 * mm, height=90 * mm))
        story.append(Spacer(1, 6))
        if visual_caption:
            story.append(Paragraph(visual_caption, styles["subtitle"]))
        story.append(PageBreak())
        story.append(Paragraph(f"Topic: {user_query}", styles["subtitle"]))
        story.append(
            Paragraph(
                f"Prepared {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                styles["subtitle"],
            )
        )
        story.append(
            HRFlowable(width="100%", thickness=1.2, lineCap="round", color=BCG_GREEN, spaceBefore=6, spaceAfter=12)
        )
    else:
        story.append(Paragraph("Strategy perspectives", styles["title"]))
        story.append(Paragraph(f"Topic: {user_query}", styles["subtitle"]))
        story.append(
            Paragraph(
                f"Prepared {datetime.now().strftime('%Y-%m-%d %H:%M')} | Strategy memorandum format",
                styles["subtitle"],
            )
        )
        story.append(
            HRFlowable(width="100%", thickness=1.2, lineCap="round", color=BCG_GREEN, spaceBefore=4, spaceAfter=12)
        )
        if visual_error:
            story.append(Paragraph(f"Visual generation note: {visual_error}", styles["subtitle"]))
            story.append(Spacer(1, 8))

    if executive_takeaways:
        story.append(Paragraph("Executive takeaways", styles["section"]))
        for takeaway in enriched_takeaways[:5]:
            story.append(Paragraph(_markdown_to_reportlab_html(takeaway), styles["bullet"], bulletText="•"))
        story.append(Spacer(1, 6))

    if chart_path and chart_path.exists():
        story.append(Paragraph("Stock impact scorecard", styles["section"]))
        story.append(Image(str(chart_path), width=160 * mm, height=70 * mm))
        story.append(Spacer(1, 6))

    if stock_calls:
        story.append(Paragraph("Prioritized stock calls", styles["section"]))
        table_data = [["Company", "Ticker", "Direction", "Confidence", "Horizon"]]
        for call in stock_calls[:10]:
            ticker_label = _normalize_ticker(str(call.get("ticker", "")))
            table_data.append(
                [
                    str(call.get("company", "")).strip(),
                    ticker_label or str(call.get("ticker", "")),
                    str(call.get("direction", "")),
                    str(call.get("confidence", "")),
                    str(call.get("time_horizon", "")),
                ]
            )
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), BCG_GREEN),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD4DC")),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))

    for section_title, section_lines in _iter_sections(report_markdown):
        story.append(Paragraph(section_title.strip(), styles["section"]))
        for line in section_lines:
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 4))
                continue
            inline_html = _markdown_to_reportlab_html(stripped)
            if stripped.startswith("### "):
                story.append(Paragraph(_markdown_to_reportlab_html(stripped[4:]), styles["subsection"]))
            elif stripped.startswith("- "):
                story.append(Paragraph(_markdown_to_reportlab_html(stripped[2:]), styles["bullet"], bulletText="•"))
            else:
                story.append(Paragraph(inline_html, styles["body"]))
        story.append(Spacer(1, 5))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
