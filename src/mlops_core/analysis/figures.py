"""Static figures for the analysis pipeline.

PNG rather than an interactive chart: these are read in the README, in the model card
and in a pull request, where a tooltip does not exist. Each figure is saved next to the
CSV of the exact table it draws, which doubles as the table view the accessibility rule
requires for the two hues that sit below 3:1 on this surface.

Colours come from the project's validated palette: fixed categorical slots (never
cycled), a blue-to-red diverging pair for signed values, a single blue for plain
magnitude. Signed charts get a zero line; nothing here ever gets a second y-axis.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # no display in a container or in CI

from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
BETTER, WORSE = "#2a78d6", "#e34948"  # the diverging pair, used for signed values
FIGURE_SIZE = (7.2, 4.2)
DPI = 160


def canvas(title: str, subtitle: str = "") -> tuple[Figure, Axes]:
    figure, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=SECONDARY, fontsize=9, va="bottom")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    return figure, ax


def value_grid(ax: Axes, axis: Literal["x", "y"] = "x") -> None:
    """A hairline grid on the value axis only: it helps reading, never competes."""
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def target_distribution_figure(table: pl.DataFrame, period: str, target: str) -> Figure:
    """Box per period from the already-computed quartiles: the shape of each period
    side by side is what shows a truncated sample, which a mean alone hides."""
    figure, ax = canvas(
        f"{target.replace('_', ' ')} by period",
        "box = quartiles, whiskers = min and max",
    )
    stats = [
        {
            "label": f"{row[period]}\n(n={row['n']})",
            "med": row["median"],
            "q1": row["q25"],
            "q3": row["q75"],
            "whislo": row["min"],
            "whishi": row["max"],
            "fliers": [],
        }
        for row in table.rows(named=True)
    ]
    boxes = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.45)
    for patch, colour in zip(boxes["boxes"], SERIES, strict=False):
        patch.set(facecolor=colour, edgecolor=colour, alpha=0.35, linewidth=1.5)
    for median, colour in zip(boxes["medians"], SERIES, strict=False):
        median.set(color=colour, linewidth=2)
    for part in ("whiskers", "caps"):
        for artist in boxes[part]:
            artist.set(color=BASELINE, linewidth=1.2)
    value_grid(ax, axis="y")
    figure.tight_layout()
    return figure


def feature_importance_figure(table: pl.DataFrame) -> Figure:
    """Permutation importance, signed: bars to the right are features the model relies
    on, bars to the left are features whose removal would *help* on the test split."""
    ranked = table.drop_nulls("permutation_importance").sort("permutation_importance")
    figure, ax = canvas(
        "What the champion actually relies on",
        "increase in error when the feature is shuffled (test split)",
    )
    colours = [BETTER if value > 0 else WORSE for value in ranked["permutation_importance"]]
    ax.barh(ranked["feature"], ranked["permutation_importance"], color=colours, height=0.62)
    ax.axvline(0, color=BASELINE, linewidth=1)
    value_grid(ax)
    ax.set_xlabel("MAE increase when shuffled", color=SECONDARY, fontsize=9)
    figure.tight_layout()
    return figure


def residual_bias_figure(table: pl.DataFrame, period: str, band_labels: list[str]) -> Figure:
    """Bias by band of the target in the most recent period: the compression a weak
    regressor shows, made visible. Only the newest period, because error on the data the
    model trained on flatters it."""
    order = {label: position for position, label in enumerate(band_labels)}
    latest = str(table[period].max())  # polars types a max() as any scalar
    bands = (
        table.filter((pl.col("kind") == "quality_band") & (pl.col(period) == latest))
        .with_columns(pl.col("level").replace_strict(order, return_dtype=pl.Int32).alias("_order"))
        .sort("_order", descending=True)
    )
    figure, ax = canvas(
        f"Where the model is wrong ({latest})",
        "mean signed error; negative = the model under-rates the item",
    )
    colours = [WORSE if value < 0 else BETTER for value in bands["bias"]]
    ax.barh(bands["level"], bands["bias"], color=colours, height=0.45)
    ax.axvline(0, color=BASELINE, linewidth=1)
    for level, bias, n in zip(bands["level"], bands["bias"], bands["n"], strict=True):
        offset = 0.05 if bias >= 0 else -0.05
        ax.text(
            bias + offset,
            level,
            f"{bias:+.2f} (n={n})",
            va="center",
            ha="left" if bias >= 0 else "right",
            color=SECONDARY,
            fontsize=9,
        )
    ax.margins(x=0.45)  # room for the value labels at both ends
    value_grid(ax)
    figure.tight_layout()
    return figure


def numeric_signal_figure(table: pl.DataFrame) -> Figure:
    """Signed correlation with the target: direction matters as much as size."""
    ranked = table.drop_nulls("correlation_with_target").sort("correlation_with_target")
    figure, ax = canvas(
        "How each numeric feature moves with the score",
        "correlation with the target across both periods",
    )
    colours = [BETTER if value > 0 else WORSE for value in ranked["correlation_with_target"]]
    ax.barh(ranked["feature"], ranked["correlation_with_target"], color=colours, height=0.62)
    ax.axvline(0, color=BASELINE, linewidth=1)
    value_grid(ax)
    figure.tight_layout()
    return figure


def render_all(
    tables: dict[str, pl.DataFrame], period: str, target: str, band_labels: list[str]
) -> dict[str, Figure]:
    """Every figure the core draws from the studies just computed.

    A study can legitimately come out empty (say, no predictions yet). That is reported
    by the pipeline and skipped here: one empty table must not take the whole run down
    with it. A domain's own figures are drawn by the domain.
    """
    drawings: list[tuple[str, str, Callable[[pl.DataFrame], Figure]]] = [
        (
            "target_distribution",
            "target_distribution",
            lambda table: target_distribution_figure(table, period, target),
        ),
        ("numeric_signal", "numeric_profile", numeric_signal_figure),
        (
            "feature_importance",
            "feature_recommendation",
            lambda table: feature_importance_figure(table),
        ),
        (
            "residual_bias",
            "residuals",
            lambda table: residual_bias_figure(table, period, band_labels),
        ),
    ]
    figures = {}
    for name, source, draw in drawings:
        table = tables.get(source)
        if table is None or table.is_empty() or _nothing_to_draw(name, table):
            logger.info("Skipping the '%s' figure: '%s' has nothing to draw", name, source)
            continue
        figures[name] = draw(table)
    return figures


def _nothing_to_draw(name: str, table: pl.DataFrame) -> bool:
    """Importance is only drawn when it was actually measured (a champion existed)."""
    if name != "feature_importance":
        return False
    return table["permutation_importance"].null_count() == table.height
