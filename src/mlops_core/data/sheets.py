"""Product sheets -> labelled values, whatever markup a shop wraps them in.

Scraped shops describe an item in one of two shapes: "Label: value" lines in a
description, or a heading followed by a paragraph on the item's page. Both come down to
the same thing, a sequence of (field, value) pairs, which is all this module reads. What
the labels *mean* is the domain's business: it passes a mapping from each label, as the
shops write it, to a field of its own.

Two traps shape the parsing:

- Labels are not always on their own line. Some sheets run them together
  ("RwandaRegion: Huye"), so a value ends at the next known label as well as at a line
  break. Hence the labels a domain does not want are still passed, mapped to None: they
  are read, so they end the value before them, and then dropped.
- One sheet can describe several things in turn - a blend lists each component with the
  same labels - so a field seen twice starts a new record instead of overwriting the
  first one.
"""

import html
import re
from collections.abc import Iterable, Mapping

_BREAKS = re.compile(r"<br\s*/?>|</(?:p|li|h\d|div|tr)>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
# A heading and the paragraph under it, allowing one wrapper element in between.
_HEADED = re.compile(
    r"<h\d[^>]*>\s*([^<]{1,60}?)\s*</h\d>\s*(?:<div[^>]*>\s*)?<p[^>]*>(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)

Pairs = list[tuple[str, str]]


def html_text(markup: str | None) -> str:
    """Markup -> plain text: one line per block element, whitespace collapsed."""
    if not markup:
        return ""
    text = html.unescape(_TAGS.sub(" ", _BREAKS.sub("\n", markup)))
    lines = (" ".join(line.split()) for line in text.splitlines())  # also folds nbsp
    return "\n".join(line for line in lines if line)


def labelled_lines(text: str, labels: Mapping[str, str | None]) -> Pairs:
    """Every "Label: value" in `text`, in reading order, as (field, value).

    Labels match whatever their case; values end at a line break or at the next label.
    """
    fields = {label.casefold(): field for label, field in labels.items()}
    # Longest first, so "Producers" is not read as "Producer" followed by "s:".
    names = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pair = re.compile(rf"({names})\s*:\s*(.*?)(?=(?:{names})\s*:|$)", re.IGNORECASE | re.MULTILINE)
    return _kept(
        (fields[match.group(1).casefold()], match.group(2)) for match in pair.finditer(text)
    )


def headed_paragraphs(markup: str | None, labels: Mapping[str, str | None]) -> Pairs:
    """Every heading that is a known label, paired with the paragraph under it.

    Headings that are not labels are skipped: a page has many, and only these are data.
    """
    fields = {label.casefold(): field for label, field in labels.items()}
    found = (
        (fields.get(html.unescape(heading).casefold()), html_text(paragraph))
        for heading, paragraph in _HEADED.findall(markup or "")
    )
    return _kept(found)


def records(pairs: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    """Group pairs into records, in order: a field seen again starts the next record."""
    grouped: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for field, value in pairs:
        if field in current:
            grouped.append(current)
            current = {}
        current[field] = value
    if current:
        grouped.append(current)
    return grouped


def _kept(pairs: Iterable[tuple[str | None, str]]) -> Pairs:
    """Drop the labels mapped to None, and labels written with nothing after them."""
    return [
        (field, " ".join(value.split()))
        for field, value in pairs
        if field is not None and value.strip()
    ]
