"""Coffee's own studies: the world market its lots come from, and the city's places.

Every domain gets the core's analyses (profiles, drift, feature evidence, residuals).
These only make sense for coffee: who grows it and who imports what they drink, what
the places in DENUE's broad "cafeterias" class actually are, how far the rule that
says so can be trusted, and how much the roasters' sheets actually say about each
coffee. Pure functions of clean tables, drawn in the core's house style.
"""

from collections.abc import Mapping

import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from domains.coffee.config import UNCLASSIFIED, MarketAnalysisConfig, ProductionConfig
from mlops_core.analysis.figures import (
    INK,
    MUTED,
    SECONDARY,
    SERIES,
    SURFACE,
    canvas,
    value_grid,
)

COFFEE = "coffee"  # the kind the whole thesis is about
# PSD counts thousands of 60 kg bags: one thousand bags is 60 tonnes.
TONNES_PER_THOUSAND_BAGS = 60.0
# What a roaster's sheet can tell about a coffee, from its origins; `processing_method`
# counts only the labels the rules understood.
SHEET_FIELDS = {
    "country": "country",
    "state": "state",
    "region": "region",
    "producer": "producer",
    "farm": "farm",
    "altitude_min_m": "altitude",
    "varieties": "varieties",
    "process": "process",
    "processing_method": "processing method",
    "species": "species",
    "sca_score": "SCA score",
}
COVERAGE_FIELDS = ["sheet", *SHEET_FIELDS.values(), "price per kg"]
ALL_SHOPS = "all"


def studies(
    clean: Mapping[str, pl.DataFrame], market: MarketAnalysisConfig, crop: ProductionConfig
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
        "production_by_state": production_by_state(clean["mexico_production"]),
        "production_crosscheck": production_crosscheck(
            clean["mexico_production"], context, crop.country
        ),
        "roaster_coverage": roaster_coverage(
            clean["roaster_coffees"], clean["roaster_origins"], clean["roaster_offers"]
        ),
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
    if not tables["production_by_state"].is_empty():
        drawn["mexico_production"] = production_figure(tables["production_by_state"])
    denue = tables["shop_kinds"].filter(pl.col("source") == "denue")
    if not denue.is_empty():
        drawn["shop_kinds"] = shop_kinds_figure(denue)
    if not tables["roaster_coverage"].is_empty():
        drawn["roaster_coverage"] = roaster_coverage_figure(tables["roaster_coverage"])
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


def production_by_state(production: pl.DataFrame) -> pl.DataFrame:
    """Where Mexico grows its coffee: each state's cherry, its share, and its price.

    The price is value over volume of the state's totals - the average a tonne actually
    fetched there - not a mean of municipal prices.
    """
    return (
        production.group_by("year", "state")
        .agg(
            pl.len().alias("municipalities"),
            pl.col("planted_ha").sum(),
            pl.col("production_t").sum(),
            pl.col("value_mxn").sum(),
        )
        .with_columns(
            (100 * pl.col("production_t") / pl.col("production_t").sum().over("year")).alias(
                "share_pct"
            ),
            pl.when(pl.col("production_t") > 0)
            .then(pl.col("value_mxn") / pl.col("production_t"))
            .alias("rural_price_mxn_per_t"),
        )
        .sort("year", "production_t", descending=[False, True])
    )


def production_crosscheck(
    production: pl.DataFrame, context: pl.DataFrame, country: str
) -> pl.DataFrame:
    """SIAP's national cherry total against PSD's green coffee for the same country.

    The two measure different things - SIAP weighs cherry as picked, PSD green coffee
    ready to export - so they should differ by a steady factor, the weight lost in
    processing. That factor is not asserted here; the table shows what it would have
    to be for the two sources to agree, for PSD's market year labelled like SIAP's year
    and for the one before (their calendars are not documented as aligned).
    """
    national = production.group_by("year").agg(pl.col("production_t").sum().alias("siap_cherry_t"))
    psd = context.filter(pl.col("country") == country).select(
        pl.col("market_year").alias("psd_market_year"),
        (pl.col("production") * TONNES_PER_THOUSAND_BAGS).alias("psd_green_t"),
    )
    pairs = pl.concat(
        [national.with_columns((pl.col("year") - lag).alias("psd_market_year")) for lag in (1, 0)]
    )
    return (
        pairs.join(psd, on="psd_market_year", how="inner")
        .with_columns((pl.col("siap_cherry_t") / pl.col("psd_green_t")).alias("cherry_per_green"))
        .select(
            pl.col("year").alias("siap_year"),
            "psd_market_year",
            "siap_cherry_t",
            "psd_green_t",
            "cherry_per_green",
        )
        .sort("siap_year", "psd_market_year")
    )


def production_figure(by_state: pl.DataFrame) -> Figure:
    """Pure magnitude, one hue, latest year: the states that grow Mexico's coffee."""
    latest = by_state.filter(pl.col("year") == by_state["year"].max())
    ranked = latest.sort("production_t")
    figure, ax = canvas(
        f"Where Mexico grows its coffee ({latest['year'][0]})",
        "tonnes of coffee cherry, SIAP closing statistics",
    )
    ax.barh(ranked["state"], ranked["production_t"], color=SERIES[0], height=0.62)
    for state, tonnes, share in zip(
        ranked["state"], ranked["production_t"], ranked["share_pct"], strict=True
    ):
        ax.text(
            tonnes,
            state,
            f"  {tonnes:,.0f} ({share:.0f}%)",
            va="center",
            color=SECONDARY,
            fontsize=9,
        )
    ax.margins(x=0.25)
    value_grid(ax)
    figure.tight_layout()
    return figure


def roaster_coverage(
    coffees: pl.DataFrame, origins: pl.DataFrame, offers: pl.DataFrame
) -> pl.DataFrame:
    """How much each shop's sheets say: per shop and field, the share of its coffees that
    give it, plus a row for all shops together.

    "sheet" is having one at all. A coffee gives a field if any of its origins does, so
    a blend counts once. The price per kilogram is counted over offers, not coffees.
    It says where the stage-3 price model can learn, and from how few shops.
    """
    understood = pl.col("processing_method").is_not_null() & (
        pl.col("processing_method") != UNCLASSIFIED
    )
    given = origins.group_by("shop", "product_id").agg(
        *[
            (understood if column == "processing_method" else pl.col(column).is_not_null())
            .any()
            .alias(field)
            for column, field in SHEET_FIELDS.items()
        ]
    )
    per_coffee = (
        coffees.select("shop", "product_id", (pl.col("origins") > 0).alias("sheet"))
        .join(given, on=["shop", "product_id"], how="left")
        .fill_null(False)
        .unpivot(index=["shop", "product_id"], variable_name="field", value_name="given")
        .drop("product_id")
    )
    per_offer = offers.select(
        "shop",
        pl.lit("price per kg").alias("field"),
        pl.col("price_mxn_per_kg").is_not_null().alias("given"),
    )
    answers = pl.concat([per_coffee, per_offer])
    answers = pl.concat([answers, answers.with_columns(shop=pl.lit(ALL_SHOPS))])
    return (
        answers.group_by("shop", "field")
        .agg(pl.len().alias("of"), pl.col("given").sum().cast(pl.Int64).alias("given"))
        .with_columns((100 * pl.col("given") / pl.col("of")).alias("share_pct"))
        .sort(
            pl.col("shop") == ALL_SHOPS,
            "shop",
            pl.col("field").cast(pl.Enum(COVERAGE_FIELDS)),
        )
    )


def roaster_coverage_figure(coverage: pl.DataFrame) -> Figure:
    """One cell per shop and field, one hue for the share: where the sheets are thin."""
    shops = coverage["shop"].unique(maintain_order=True).to_list()
    coffees = dict(coverage.filter(pl.col("field") == "sheet").select("shop", "of").iter_rows())
    grid = coverage.pivot(on="shop", index="field", values="share_pct")
    shares = grid.select(shops).to_numpy()
    figure, ax = canvas(
        "What the roasters' sheets say about each coffee",
        "Share of each shop's coffees whose sheet gives the field (price: of its offers)",
    )
    ax.imshow(
        shares,
        cmap=LinearSegmentedColormap.from_list("share", [SURFACE, SERIES[0]]),
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    for row in range(grid.height):
        for column in range(len(shops)):
            share = shares[row, column]
            ax.text(
                column,
                row,
                f"{share:.0f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color=SURFACE if share >= 60 else INK,
            )
    ax.set_xticks(range(len(shops)), [f"{shop}\n{coffees[shop]} coffees" for shop in shops])
    ax.set_yticks(range(grid.height), grid["field"].to_list())
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure
