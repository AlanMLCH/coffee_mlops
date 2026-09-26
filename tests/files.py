"""Files written for tests, byte by byte: a workbook and a PDF page of text.

A recorded download would be a third party's file in the repository; these are written
in the shape the real ones have, from the few cells or lines a test needs, so what a
test reads is in the test.
"""

import io
import zipfile
from collections.abc import Sequence
from xml.sax.saxutils import escape

Cell = str | float | None

_XML = '<?xml version="1.0" encoding="UTF-8"?>'
_PACKAGE = "http://schemas.openxmlformats.org/package/2006"
_OPC = "application/vnd.openxmlformats-package"
_SHEETML = "application/vnd.openxmlformats-officedocument.spreadsheetml"
_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def xlsx(sheet: str, rows: Sequence[Sequence[Cell]]) -> bytes:
    """A workbook with one sheet: text as inline strings, numbers as numbers, and an
    empty cell where a row says None."""
    cells = []
    for r, row in enumerate(rows, start=1):
        written = []
        for c, value in enumerate(row):
            ref = f"{chr(ord('A') + c)}{r}"
            if isinstance(value, str):
                written.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            elif value is not None:
                written.append(f'<c r="{ref}"><v>{value}</v></c>')
        cells.append(f'<row r="{r}">{"".join(written)}</row>')
    parts = {
        "[Content_Types].xml": (
            f'{_XML}<Types xmlns="{_PACKAGE}/content-types">'
            f'<Default Extension="rels" ContentType="{_OPC}.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f'<Override PartName="/xl/workbook.xml" ContentType="{_SHEETML}.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            f'ContentType="{_SHEETML}.worksheet+xml"/></Types>'
        ),
        "_rels/.rels": _relationship("officeDocument", "xl/workbook.xml"),
        "xl/workbook.xml": (
            f'{_XML}<workbook xmlns="{_MAIN}" xmlns:r="{_RELATIONSHIPS}">'
            f'<sheets><sheet name="{escape(sheet)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": _relationship("worksheet", "worksheets/sheet1.xml"),
        "xl/worksheets/sheet1.xml": (
            f'{_XML}<worksheet xmlns="{_MAIN}"><sheetData>{"".join(cells)}</sheetData></worksheet>'
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def pdf(lines: Sequence[str]) -> bytes:
    """A one-page PDF whose text is `lines`, one under the other, in Helvetica."""
    text = "".join(
        f"BT /F1 10 Tf 40 {800 - 14 * n} Td ({_pdf_text(line)}) Tj ET\n"
        for n, line in enumerate(lines)
    ).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%bendstream" % (len(text), text),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%b\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _relationship(kind: str, target: str) -> str:
    return (
        f'{_XML}<Relationships xmlns="{_PACKAGE}/relationships">'
        f'<Relationship Id="rId1" Type="{_RELATIONSHIPS}/{kind}" Target="{target}"/>'
        "</Relationships>"
    )


def _pdf_text(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
