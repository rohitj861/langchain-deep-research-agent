"""Convert the Markdown report into a downloadable PDF (fpdf2, pure Python)."""

import re
from pathlib import Path

import markdown
from fpdf import FPDF

# Unicode TTF fonts. On Streamlit Cloud, `packages.txt` installs fonts-dejavu-core.
_FONT_DIRS = [
    Path(__file__).parent / "fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("C:/Windows/Fonts"),
]
_FONT_SETS = [
    ("DejaVu", {"": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf", "I": "DejaVuSans-Oblique.ttf", "BI": "DejaVuSans-BoldOblique.ttf"}),
    ("Arial", {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf", "BI": "arialbi.ttf"}),
]

# Fallback when no Unicode font exists: map common typography to Latin-1.
_LATIN1_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u2026": "...", "\u2022": "*", "\u00a0": " ", "\u2192": "->", "\u2264": "<=", "\u2265": ">=",
    "\u2248": "~", "\u2713": "Y", "\u2717": "N", "\u2715": "N", "\u2714": "Y",
})


def _register_unicode_font(pdf: FPDF) -> str | None:
    for family, styles in _FONT_SETS:
        for d in _FONT_DIRS:
            if all((d / f).exists() for f in styles.values()):
                for style, f in styles.items():
                    pdf.add_font(family, style, str(d / f))
                return family
    return None


def _to_latin1(text: str) -> str:
    return text.translate(_LATIN1_MAP).encode("latin-1", "replace").decode("latin-1")


def _clean_html(html: str) -> str:
    # fpdf2's HTML renderer does not support these; keep their text content.
    html = re.sub(r"</?(span|div|colgroup|col)[^>]*>", "", html)
    # Inline code inside tables/paragraphs renders fine as plain text.
    html = re.sub(r"</?code>", "", html)
    # Table cell alignment attributes from the markdown tables extension.
    html = re.sub(r'\s(style|align)="[^"]*"', "", html)
    return html


def markdown_to_pdf(md_text: str, title: str = "Research Report") -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)
    pdf.set_title(title)
    pdf.set_author("LangChain Deep Research Agent")

    family = _register_unicode_font(pdf)
    if family is None:
        family = "helvetica"
        md_text = _to_latin1(md_text)

    pdf.add_page()
    pdf.set_font(family, size=11)

    html = markdown.markdown(md_text, extensions=["tables", "sane_lists", "fenced_code"])
    html = _clean_html(html)

    try:
        pdf.write_html(
            html,
            font_family=family,
            table_line_separators=True,
            tag_styles=None,
        )
    except Exception:
        # Last-resort: render as plain text so the user still gets a PDF.
        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        fam = _register_unicode_font(pdf) or "helvetica"
        pdf.set_font(fam, size=10)
        body = md_text if fam != "helvetica" else _to_latin1(md_text)
        pdf.multi_cell(0, 5, body)

    return bytes(pdf.output())
