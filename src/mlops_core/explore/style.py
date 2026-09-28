"""The explorer's look: one palette for the page, its charts and its map.

Warm, and deliberately not a domain's colours: the core names no domain. `mlops explore`
writes the theme to a file and hands Streamlit its path (`write_theme`), so no config
file has to sit in whatever directory the app is started from; the map takes the same
colours as RGB, since deck.gl does not read the page's theme.
"""

from pathlib import Path

Rgb = tuple[int, int, int]

ACCENT = "#B5653A"  # burnt sienna: buttons, links, the first series
INK = "#2B1D14"
PAPER = "#FFFDF9"
CREMA = "#F4ECE3"  # the second background: cards, the sidebar, code
LINE = "#E6D8C8"
# The categorical series, in order: sienna, blue, green, amber, plum, stone.
SERIES = ["#B5653A", "#2A78D6", "#1B966E", "#E6A000", "#8E5BA8", "#898781"]
# A number from little to much: cream to dark brown.
SEQUENTIAL = ["#F6E8D6", "#EBCFAE", "#DDB286", "#CC9160", "#B5653A", "#93492A", "#6E3418"]

THEME: dict[str, str | list[str]] = {
    "base": "light",
    "primaryColor": ACCENT,
    "backgroundColor": PAPER,
    "secondaryBackgroundColor": CREMA,
    "textColor": INK,
    "borderColor": LINE,
    "headingFont": "serif",
    "baseRadius": "0.6rem",
    "chartCategoricalColors": SERIES,
    "chartSequentialColors": SEQUENTIAL,
}

# What the theme cannot say: the header band, the cards' hover, the caption under a map.
CSS = f"""
<style>
.explore-hero {{
    background: linear-gradient(120deg, {INK} 0%, #5A3522 55%, {ACCENT} 100%);
    color: {PAPER}; border-radius: 1rem; padding: 1.6rem 2rem 1.3rem; margin-bottom: 0.8rem;
}}
.explore-hero h1 {{ color: {PAPER}; font-family: serif; margin: 0 0 0.4rem; font-size: 2.3rem; }}
.explore-hero p {{ color: #F3E6D8; margin: 0; font-size: 1.02rem; max-width: 62rem; }}
.explore-note {{
    background: {CREMA}; border-left: 4px solid {ACCENT}; border-radius: 0.5rem;
    padding: 0.7rem 1rem; margin: 0.3rem 0 0.8rem; font-size: 0.93rem;
}}
.explore-ramp {{
    height: 0.7rem; border-radius: 0.35rem; margin: 0.2rem 0 0.1rem;
    background: linear-gradient(90deg, {", ".join(SEQUENTIAL)});
}}
.explore-legend {{ display: flex; justify-content: space-between; font-size: 0.8rem; }}
.explore-chip {{
    display: inline-block; border-radius: 999px; padding: 0.05rem 0.6rem; margin-right: 0.3rem;
    background: {CREMA}; border: 1px solid {LINE}; font-size: 0.78rem;
}}
</style>
"""


def rgb(color: str) -> Rgb:
    """ "#B5653A" -> (181, 101, 58)."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def write_theme(path: Path) -> Path:
    """The theme as a TOML theme file, which Streamlit takes as `--theme.base <path>`.

    A file, not one option per setting: on the command line a list arrives as a string
    (`chartCategoricalColors = "[...]"`, checked with `streamlit config show`), and the
    series colours are lists."""

    def written(value: str | list[str]) -> str:
        return (
            "[" + ", ".join(f'"{v}"' for v in value) + "]"
            if isinstance(value, list)
            else f'"{value}"'
        )

    lines = ["[theme]", *(f"{name} = {written(value)}" for name, value in THEME.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
