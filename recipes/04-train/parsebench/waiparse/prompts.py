"""Prompts for the three agent stages. Each output shape is chosen to match what a
ParseBench scorer reads (the scorer notes are in this recipe's README)."""

CATEGORIES = [
    "Title",
    "Section-header",
    "Text",
    "List-item",
    "Caption",
    "Footnote",
    "Formula",
    "Table",
    "Picture",
    "Page-header",
    "Page-footer",
]

LAYOUT = f"""Output the layout of this document page as a JSON array, one object per layout element, in human reading order:
{{"bbox": [x1, y1, x2, y2], "category": "<category>", "text": "<content>"}}

- bbox: normalized 0-1000 coordinates, tight around the element. One box per paragraph, list item, heading, caption, footnote. Never merge two paragraphs into one box.
- category: one of {CATEGORIES}. Title = the document or page title; Section-header = any other heading. Running headers/page numbers at the top are Page-header, at the bottom Page-footer.
- text: the exact original text, no translation, no paraphrase, keep line content complete.
  - Mark styling only where it is visibly present: **bold**, ~~strikethrough~~, <sup>superscript</sup> (footnote markers too), <sub>subscript</sub>. Do not bold headings.
  - Formula: LaTeX without delimiters.
  - Table and Picture: leave text as "" (they are transcribed separately).
Return ONLY the JSON array."""

TABLE = """Transcribe this table as HTML.
- <table>, then <thead> holding every header row, where every header cell is <th> (including an empty top-left corner <th></th>), then <tbody> with <td> cells.
- Multi-level headers stay as separate <thead> rows; merged cells use rowspan/colspan exactly as drawn.
- Copy every cell exactly as printed: numbers, signs, parentheses, currency, %, dashes. Empty cells are empty <td></td>.
- Footnote markers as <sup>..</sup>; bold only if visibly bold.
- If the table has a title above it inside the image, put it in <caption>.
Return ONLY the HTML table, no markdown, no commentary."""

CHART = """Is this image a chart or graph (bar, line, pie, area, scatter, stacked, combo, map with values)?
If NO (photo, logo, diagram, icon, text), reply exactly: NONE

If YES, extract the chart's underlying data as an HTML table:
<table><caption>chart title as printed (omit if none)</caption>
<thead><tr><th>category axis label</th><th>series name 1</th><th>series name 2</th>...</tr></thead>
<tbody><tr><td>category 1</td><td>value</td>...</tr>...</tbody></table>
- Series names from the legend exactly as printed; with a single unnamed series use the value-axis title or chart title as the header.
- Categories (x-axis ticks, pie slices, bar labels) exactly as printed, one row each, in chart order.
- Each value cell holds ONE number exactly as printed on the chart (keep decimals and minus signs; drop units, %, currency, and k/M/B suffixes). If values are not printed, estimate from the axis gridlines as precisely as you can.
- Several charts in the image: one table per chart.
Return ONLY the HTML table(s) or NONE."""

CHART_PAGE = """Find every chart or graph on this page (bar, line, pie, area, scatter, stacked, combo, dot, map with values). Ignore photos, logos, diagrams and ordinary tables.
If there are none, reply exactly: NONE

For EACH chart, in reading order, output its underlying data as one HTML table:
<table><caption>the chart's own title as printed (e.g. "Figure 2.2. Total merger notifications")</caption>
<thead><tr><th>category axis label</th><th>series name 1</th><th>series name 2</th>...</tr></thead>
<tbody><tr><td>category 1</td><td>value</td>...</tr>...</tbody></table>
- Series names from the legend exactly as printed (the legend may sit below or beside the plot); with a single unnamed series use the value-axis title or chart title as the header.
- Categories (x-axis ticks, years, countries, pie slices, bar labels) exactly as printed, one row each, in chart order.
- Each value cell holds ONE number exactly as printed on the chart (keep decimals and minus signs; drop units, %, currency, and k/M/B suffixes). If values are not printed, estimate from the axis gridlines as precisely as you can.
- Small multiples (one panel per country/segment): one table per panel, the panel title as caption.
Return ONLY the HTML tables or NONE."""

STYLE = """Transcribe the text in this image exactly, keeping its line breaks between paragraphs or list items.
Mark styling ONLY where it is visibly present in the image:
- **bold** for bold or heavy type (whole bold lines too)
- ~~strikethrough~~ for struck-through text
- <sup>x</sup> for raised text: footnote markers, citation numbers like [12] or ¹, ordinals, exponents
- <sub>x</sub> for lowered text: chemical formulas, indices
Never add styling that is not visible. No italics markup. No commentary, no code fences."""
