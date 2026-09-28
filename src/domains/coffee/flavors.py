"""Tasting notes: what the roasters say their coffees taste of, in the SCA's categories.

A shop's description tells a story - the producer, the farm, the harvest - and, in one
or two sentences, what the cup tastes of: "Un café con notas a chocolate y frambuesa".
`roaster_flavors` keeps one row per coffee and note, each placed in a flavour category
of the SCA's descriptive assessment (SCA-103, which is in the corpus), so a bag bought
in Mexico City in 2026 can be read in the vocabulary a cupper uses.

A lexicon, not a model, and on purpose: every row can be traced to the word that put it
there, the rules can be read aloud, and there are 167 coffees, not a corpus. The rules
are `cleaning.tasting_notes` in the domain config. The reading is kept narrow, because a
wrong note is worse than a missing one:

- A note counts only after a cue ("notas a", "sabe a", "aroma", "en taza") and before
  the end of its sentence. "Este compromiso con la tierra" is not earthy.
- A cue right after a negation opens nothing ("sin llegar a los sabores fermentados"),
  and a negation inside a list of notes ends it.
- A capitalised word mid-sentence is a name, not a note: Juan Carlos Flores, Finca
  Santa Rosa.
- A word used in another sense is not a note: in coffee, "cereza" is also the fruit that
  is picked ("la selección de cerezas maduras").
- Only the shops that write their notes in the title have their titles read, after the
  hyphen: Cucurucho's "Chiapas- Caramelo, avellana y chocolate".
"""

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

import polars as pl

from domains.coffee.config import TastingNotesConfig

FLAVORS = pl.Schema(
    {
        "coffee_id": pl.String,
        "shop": pl.String,
        "note": pl.String,  # as the shop wrote it, in Spanish
        "note_en": pl.String,
        "category": pl.String,
        "subcategory": pl.String,
        "source": pl.String,  # where the shop wrote it: "title" or "description"
    }
)
TITLE, DESCRIPTION = "title", "description"
# Words inside a multi-word note that never take a plural: "fruta con hueso".
_JOINERS = {"a", "al", "con", "de", "del", "el", "en", "la", "las", "los", "y"}
_SENTENCE = re.compile(r"[^.!?;\n]+")
_WORD = re.compile(r"\w+")
# A capitalised note after one of these starts an item of a list, so it is not a name.
_LIST_MARKS = ":-,;(/"
# How many words a negation reaches ahead: "ni llegar a los sabores".
_NEGATION_REACH = 3


@dataclass(frozen=True)
class Note:
    """One entry of the lexicon."""

    note: str
    note_en: str
    category: str
    subcategory: str | None


def folded(text: str) -> str:
    """Lower-case, accents dropped, one character for each character of `text`: positions
    found in the folded text are positions in the original, which the capitalisation
    rule reads. (NFKD alone would not do: "ﬀ" becomes two letters.)"""
    return "".join(
        (unicodedata.normalize("NFKD", c)[0] if c.isalpha() else c).lower()
        for c in unicodedata.normalize("NFC", text)
    )


def inflected(note: str) -> str:
    """A pattern for a note in its plural, and the last word in its other gender:
    "fruto rojo" -> "frutos rojos", "caramelizado" -> "caramelizada"."""
    words = folded(note).split()
    parts = []
    for position, word in enumerate(words):
        stem = re.escape(word)
        if word in _JOINERS:
            parts.append(stem)
        elif position == len(words) - 1 and word[-1] in "oa" and len(word) > 2:
            parts.append(f"{re.escape(word[:-1])}(?:o|a)s?")
        elif word[-1] in "aeiou":
            parts.append(f"{stem}s?")
        else:
            parts.append(f"{stem}(?:es|s)?")
    return r"\s+".join(parts)


class Lexicon:
    """The notes of `rules`, compiled once: one alternation, the longest notes first, so
    "nuez moscada" is nutmeg and not a walnut."""

    def __init__(self, rules: TastingNotesConfig) -> None:
        self.rules = rules
        entries = [
            Note(note, english, group.category, group.subcategory)
            for group in rules.groups
            for note, english in group.notes.items()
        ]
        entries.sort(key=lambda entry: -len(entry.note))
        self._patterns = [(re.compile(rf"{inflected(e.note)}"), e) for e in entries]
        alternation = "|".join(inflected(entry.note) for entry in entries)
        self._notes = re.compile(rf"\b(?:{alternation})\b")
        self._cues = re.compile(rf"\b(?:{_alternation(rules.cues)})\b")
        self._negation = re.compile(rf"\b(?:{_alternation(rules.negations)})\b")
        reach = rf"(?:\w+\W+){{0,{_NEGATION_REACH}}}"
        self._negated = re.compile(rf"\b(?:{_alternation(rules.negations)})\W+{reach}$")
        self._other = [re.compile(pattern) for pattern in rules.other_meanings]

    def entry(self, words: str) -> Note:
        """The lexicon's entry for a match of the alternation."""
        for pattern, entry in self._patterns:
            if pattern.fullmatch(words):
                return entry
        raise KeyError(words)  # pragma: no cover - every match comes from one of them

    def notes(self, text: str, source: str, whole: bool = False) -> Iterator[tuple[Note, str]]:
        """The notes `text` names, each with `source`. `whole`: the text is itself a list
        of notes (a title's part after the hyphen), no cue needed."""
        text = unicodedata.normalize("NFC", text)
        low = folded(text)
        spans = [(0, len(low))] if whole else list(self._windows(low))
        taken = [match.span() for pattern in self._other for match in pattern.finditer(low)]
        for start, end in spans:
            for match in self._notes.finditer(low, start, end):
                if any(match.start() < stop and begin < match.end() for begin, stop in taken):
                    continue  # a word in its other sense
                if _is_name(text, start, match.start()):
                    continue
                yield self.entry(match.group()), source

    def _windows(self, low: str) -> Iterator[tuple[int, int]]:
        """Each stretch of a sentence from a cue to the sentence's end, or to the first
        negation after the cue."""
        for sentence in _SENTENCE.finditer(low):
            begin, end = sentence.span()
            for cue in self._cues.finditer(low, begin, end):
                if self._negated.search(low[begin : cue.start()]):
                    continue
                negation = self._negation.search(low, cue.end(), end)
                yield cue.end(), negation.start() if negation else end


def tasting_notes(coffees: pl.DataFrame, rules: TastingNotesConfig) -> pl.DataFrame:
    """`roaster_coffees` -> one row per coffee and note it tastes of, as its shop says.

    A note is kept once per coffee, whatever its spelling: "jamaica" and "hibisco" are
    both hibiscus. A coffee whose description names no note has no rows; the share of
    coffees with notes is in `analysis.roaster_coverage`.
    """
    lexicon = Lexicon(rules)
    rows = []
    for coffee in coffees.iter_rows(named=True):
        found = list(lexicon.notes(coffee["description"] or "", DESCRIPTION))
        if coffee["shop"] in rules.title_notes and "-" in coffee["title"]:
            after = coffee["title"].split("-", 1)[1]
            found = list(lexicon.notes(after, TITLE, whole=True)) + found
        rows += [
            {
                "coffee_id": coffee["coffee_id"],
                "shop": coffee["shop"],
                "note": entry.note,
                "note_en": entry.note_en,
                "category": entry.category,
                "subcategory": entry.subcategory,
                "source": source,
            }
            for entry, source in found
        ]
    return (
        pl.DataFrame(rows, schema=FLAVORS)
        .unique(["coffee_id", "note_en"], keep="first", maintain_order=True)
        .sort("coffee_id", "category", "note_en")
    )


def _alternation(words: list[str]) -> str:
    """Words as one alternation, the longest first ("en taza" before a shorter cue)."""
    return "|".join(re.escape(folded(word)) for word in sorted(words, key=len, reverse=True))


def _is_name(text: str, window: int, at: int) -> bool:
    """A capitalised word that does not start the window or an item of a list: a name."""
    before = text[window:at].rstrip()
    return text[at].isupper() and bool(before) and before[-1] not in _LIST_MARKS
