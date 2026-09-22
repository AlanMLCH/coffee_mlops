"""Coffee's own studies: the world market its lots come from, and the city's places.

Every domain gets the core's analyses (profiles, drift, feature evidence, residuals).
These only make sense for coffee: who grows it and who imports what they drink, what
the places in DENUE's broad "cafeterias" class actually are, and how far the rule that
says so can be trusted. Pure functions of clean tables, drawn in the core's house style.
"""

from collections.abc import Mapping

import polars as pl
from matplotlib.figure import Figure

from domains.coffee.config import UNCLASSIFIED, MarketAnalysisConfig
from mlops_core.analysis.figures import MUTED, SECONDARY, SERIES, canvas, value_grid

COFFEE = "coffee"  # the kind the whole thesis is about


def studies(
    clean: Mapping[str, pl.DataFrame], market: MarketAnalysisConfig
) -> dict[str, pl.DataFrame]:
    """The domain's studies: the world market, and the city's places."""
    context, shops = clean["market_context"], clean["coffee_shops"]
    agreement = kind_agreement(shops)
    return {
        "market_summary": market_summary(context, market.market_year, market.top_countries),
        "market_history": market_history(context, market.spotlight_country, market.history_since),
        "shop_kinds": shop_kinds(shops),
        "kind_agreement": agreement,
        "kind_scores": kind_scores(agreement),
    }


def figures(tables: Mapping[str, pl.DataFrame], market: MarketAnalysisConfig) -> dict[str, Figure]:
    """One figure per market study that has rows to draw."""
    drawn = {}
    if not tables["market_summary"].is_empty():
        drawn["market_share"] = market_share_figure(tables["market_summary"], market.market_year)
    if not tables["market_history"].is_empty():
        drawn["market_history"] = market_history_figure(
            tables["market_history"], market.spotlight_country
        )
    denue = tables["shop_kinds"].filter(pl.col("source") == "denue")
    if not denue.is_empty():
        drawn["shop_kinds"] = shop_kinds_figure(denue)
    return drawn


def shop_kinds(shops: pl.DataFrame) -> pl.DataFrame:
    """How many places of each kind each register holds, and what share of it that is."""
    return (
        shops.group_by("source", "kind")
        .agg(pl.len().alias("places"))
        .with_columns(
            (100 * pl.col("places") / pl.col("places").sum().over("source")).alias("share_pct")
        )
        .sort("source", "places", descending=[False, True])
    )


def kind_agreement(shops: pl.DataFrame) -> pl.DataFrame:
    """On the places both registers list: what DENUE's name says against OSM's tag.

    OSM's `amenity` tag was set by a mapper who stood in front of the place, and it is
    independent of how DENUE spells the name, so it can score the name rule. The sample
    is not the city, though: a place both registers list is usually named, mapped, and
    central, so read the scores as how the rule does where it can be checked.
    """
    denue = shops.filter((pl.col("source") == "denue") & pl.col("matched_shop_id").is_not_null())
    twins = shops.select(
        pl.col("shop_id").alias("matched_shop_id"), pl.col("kind").alias("osm_tag")
    )
    return (
        denue.join(twins, on="matched_shop_id", how="inner")
        .group_by(pl.col("kind").alias("name_rule"), "osm_tag")
        .agg(pl.len().alias("pairs"))
        .sort("pairs", "name_rule", descending=[True, False])
    )


def kind_scores(agreement: pl.DataFrame) -> pl.DataFrame:
    """Precision and recall of the name rule for coffee, with the counts behind each.

    Precision: of the places the rule calls coffee, how many OSM calls coffee too.
    Recall: of the places OSM calls coffee, how many the rule found.
    """
    said = agreement.filter(pl.col("name_rule") == COFFEE)
    tagged = agreement.filter(pl.col("osm_tag") == COFFEE)
    both = agreement.filter((pl.col("name_rule") == COFFEE) & (pl.col("osm_tag") == COFFEE))
    hits = int(both["pairs"].sum())
    rows = [("precision", said), ("recall", tagged)]
    return pl.DataFrame(
        {
            "metric": [name for name, _ in rows],
            "value": [hits / int(base["pairs"].sum()) if base.height else None for _, base in rows],
            "hits": [hits, hits],
            "of": [int(base["pairs"].sum()) for _, base in rows],
        },
        schema={"metric": pl.String, "value": pl.Float64, "hits": pl.Int64, "of": pl.Int64},
    )


def shop_kinds_figure(denue: pl.DataFrame) -> Figure:
    """Pure magnitude, one hue: what DENUE's "cafeterias" class is actually made of."""
    ranked = denue.sort("places")
    figure, ax = canvas(
        "What DENUE's cafeterias class holds",
        "SCIAN 722515 in Mexico City, by what each name says",
    )
    # Coffee in the accent, the known other kinds in one hue, and "unclassified" in
    # grey: it is not "not coffee" but "the name does not say", and it holds the cafes
    # the rule misses.
    palette = {COFFEE: SERIES[0], UNCLASSIFIED: MUTED}
    colours = [palette.get(kind, SERIES[3]) for kind in ranked["kind"]]
    ax.barh(ranked["kind"].str.replace("_", " "), ranked["places"], color=colours, height=0.62)
    for kind, places, share in zip(
        ranked["kind"].str.replace("_", " "), ranked["places"], ranked["share_pct"], strict=True
    ):
        ax.text(
            places, kind, f"  {places:,} ({share:.0f}%)", va="center", color=SECONDARY, fontsize=9
        )
    ax.margins(x=0.2)
    value_grid(ax)
    figure.tight_layout()
    return figure


def market_summary(context: pl.DataFrame, year: int, top: int) -> pl.DataFrame:
    """Who produces the world's coffee in one market year, and what they do with it."""
    producing = context.filter((pl.col("market_year") == year) & (pl.col("production") > 0))
    return (
        producing.select(
            "country",
            "production",
            (100 * pl.col("production") / pl.col("production").sum()).alias("world_share_pct"),
            (pl.col("exports") / pl.col("production")).alias("export_ratio"),
            "domestic_consumption",
            (pl.col("imports") / pl.col("domestic_consumption")).alias("imported_share_of_use"),
        )
        .sort("production", descending=True)
        .head(top)
    )


def market_history(context: pl.DataFrame, country: str, since: int) -> pl.DataFrame:
    """One country through time: production, what it exports, and what it imports to
    drink. For Mexico those three lines are the whole story of the domestic market."""
    return (
        context.filter((pl.col("country") == country) & (pl.col("market_year") >= since))
        .select(
            "market_year",
            "production",
            "exports",
            "domestic_consumption",
            "imports",
            (100 * pl.col("imports") / pl.col("domestic_consumption")).alias(
                "imported_share_of_use_pct"
            ),
        )
        .sort("market_year")
    )


def market_share_figure(table: pl.DataFrame, year: int) -> Figure:
    """Pure magnitude, so one hue and a ranked bar: who grows the world's coffee."""
    ranked = table.sort("production")
    figure, ax = canvas(
        f"World coffee production, market year {year}",
        "thousands of 60 kg bags",
    )
    ax.barh(ranked["country"], ranked["production"], color=SERIES[0], height=0.62)
    for country, production, share in zip(
        ranked["country"], ranked["production"], ranked["world_share_pct"], strict=True
    ):
        ax.text(production, country, f"  {share:.1f}%", va="center", color=SECONDARY, fontsize=9)
    ax.margins(x=0.12)
    value_grid(ax)
    figure.tight_layout()
    return figure


def market_history_figure(table: pl.DataFrame, country: str) -> Figure:
    """Four series in one unit, so one axis. Labelled at the line end rather than in a
    legend box, which also supplies the relief the low-contrast hues require."""
    figure, ax = canvas(
        f"{country}: production, trade and consumption",
        "thousands of 60 kg bags",
    )
    years = table["market_year"].to_list()
    series = ["production", "exports", "domestic_consumption", "imports"]
    for name, colour in zip(series, SERIES, strict=True):
        values = table[name].to_list()
        ax.plot(years, values, color=colour, linewidth=2, marker="o", markersize=4)
        ax.text(
            years[-1],
            values[-1],
            f"  {name.replace('_', ' ')}",
            color=colour,
            fontsize=9,
            va="center",
        )
    # Legend below the plot: the lines already carry their name at the end, and a box
    # inside the axes would sit on top of the data.
    ax.legend(
        [name.replace("_", " ") for name in series],
        frameon=False,
        fontsize=9,
        labelcolor=SECONDARY,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=4,
    )
    # Ticks on the years that exist, and room on the right for the end labels.
    ax.set_xticks(years[:: max(len(years) // 6, 1)])
    ax.set_xlim(min(years) - 0.3, max(years) + (max(years) - min(years)) * 0.28)
    value_grid(ax, axis="y")
    figure.tight_layout()
    return figure
