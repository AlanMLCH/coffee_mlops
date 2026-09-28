"""Coffee's own studies: the world market its lots come from, and the city's places.

Every domain gets the core's analyses (profiles, drift, feature evidence, residuals).
These only make sense for coffee: who grows it and who imports what they drink, what
the places in DENUE's broad "cafeterias" class actually are, how far the rule that
says so can be trusted, how much the roasters' sheets actually say about each
coffee and what they say it tastes of, and what a kilogram costs from the farm gate
and the port to a supermarket's shelf and a roaster's shop. Pure functions of clean
tables, drawn in the core's house style.
"""

from collections.abc import Mapping
from datetime import timedelta

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

from domains.coffee.config import (
    UNCLASSIFIED,
    ConsumerPricesConfig,
    MarketAnalysisConfig,
    ProductionConfig,
)
from domains.coffee.prices import CENTS_PER_LB_PER_USD_PER_KG
from domains.coffee.roaster_sheets import fold
from mlops_core.analysis.figures import (
    INK,
    MUTED,
    SECONDARY,
    SERIES,
    SURFACE,
    canvas,
    value_grid,
)
from mlops_core.stats import bootstrap_means

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
TASTING_NOTES = "tasting notes"
COVERAGE_FIELDS = ["sheet", *SHEET_FIELDS.values(), TASTING_NOTES, "price per kg"]
ALL_SHOPS = "all"
# The numbers of clusters the flavour profiles are tried with.
CLUSTER_COUNTS = range(2, 7)
CLUSTER_SEED = 0
PRICE_RESAMPLES = 5000
PRICE_SEED = 7
NATIONAL, CITY = "national", "city"  # the two scopes of the shelf prices
# The shelf lines drawn through time: the plain and the sweetened of each product.
DRAWN_LINES = ["ground", "ground, sweetened", "instant", "instant, sweetened"]
# The ICO group Mexico's Arabica is priced in: green coffee's step of the ladder.
GREEN_STEP = "other_milds"
GREEN_LABELS = {"other_milds": "other mild Arabicas", "robustas": "Robustas"}


def studies(
    clean: Mapping[str, pl.DataFrame],
    market: MarketAnalysisConfig,
    crop: ProductionConfig,
    shelves: ConsumerPricesConfig,
    states: Mapping[str, str],
    home_country: str,
    min_rows: int,
) -> dict[str, pl.DataFrame]:
    """The domain's studies: the world market, the city's places, what the roasters say
    their coffees taste of, and the price of a kilogram from the farm to the shelf.
    `states` maps the domain's spelling of a state (folded) to SIAP's; `home_country` is
    the one the roasters' own coffees are profiled against the rest of the world; a group
    of fewer than `min_rows` coffees is not reported."""
    context, shops = clean["market_context"], clean["coffee_shops"]
    prices, production = clean["consumer_prices"], clean["mexico_production"]
    flavors, origins = clean["roaster_flavors"], clean["roaster_origins"]
    agreement = kind_agreement(shops)
    green = green_coffee_in_pesos(clean["price_indicators"], clean["exchange_rates"])
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
            clean["roaster_coffees"], origins, clean["roaster_offers"], flavors
        ),
        "flavor_profiles": flavor_profiles(flavors, origins, home_country, min_rows),
        "flavor_prices": flavor_prices(flavors, clean["roaster_offers"], min_rows),
        "flavor_clusters": flavor_clusters(flavors),
        "consumer_prices_by_fortnight": consumer_prices_by_fortnight(prices, shelves.city),
        "consumer_prices_by_borough": consumer_prices_by_borough(prices),
        "consumer_prices_by_state": consumer_prices_by_state(prices, production, states),
        "green_coffee_in_pesos": green,
        "price_ladder": price_ladder(
            prices, clean["roaster_offers"], production, green, shelves.city
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
    if not tables["flavor_profiles"].is_empty():
        drawn["flavor_profiles"] = flavor_profiles_figure(tables["flavor_profiles"])
    if not tables["price_ladder"].is_empty():
        drawn["price_ladder"] = price_ladder_figure(tables["price_ladder"])
    national = tables["consumer_prices_by_fortnight"].filter(pl.col("scope") == NATIONAL)
    if not national.is_empty():
        drawn["consumer_prices"] = consumer_prices_figure(national)
    if not tables["green_coffee_in_pesos"].is_empty():
        drawn["green_coffee_pesos"] = green_coffee_figure(tables["green_coffee_in_pesos"])
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
    coffees: pl.DataFrame, origins: pl.DataFrame, offers: pl.DataFrame, flavors: pl.DataFrame
) -> pl.DataFrame:
    """How much each shop's sheets say: per shop and field, the share of its coffees that
    give it, plus a row for all shops together.

    "sheet" is having one at all. A coffee gives a field if any of its origins does, so
    a blend counts once; it gives tasting notes if its description names one. The price
    per kilogram is counted over offers, not coffees. It says where the stage-3 price
    model can learn, and from how few shops.
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
    noted = flavors.select("coffee_id").unique().with_columns(pl.lit(True).alias(TASTING_NOTES))
    per_coffee = (
        coffees.select("shop", "product_id", "coffee_id", (pl.col("origins") > 0).alias("sheet"))
        .join(given, on=["shop", "product_id"], how="left")
        .join(noted, on="coffee_id", how="left")
        .drop("coffee_id")
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
    _share_grid(
        ax,
        shares,
        [f"{shop}\n{coffees[shop]} coffees" for shop in shops],
        grid["field"].to_list(),
    )
    figure.tight_layout()
    return figure


def _share_grid(ax: Axes, shares: np.ndarray, columns: list[str], rows: list[str]) -> None:
    """Percentages as a grid of cells, one hue from empty to full, each one labelled."""
    ax.imshow(
        shares,
        cmap=LinearSegmentedColormap.from_list("share", [SURFACE, SERIES[0]]),
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    for row in range(len(rows)):
        for column in range(len(columns)):
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
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def flavor_profiles(
    flavors: pl.DataFrame, origins: pl.DataFrame, home_country: str, min_rows: int
) -> pl.DataFrame:
    """What each group of coffees tastes of, as its roasters describe it: per group, the
    share of its coffees whose description names a note of each flavour category.

    Only coffees with notes count, so a shop that writes few does not look bland. The
    groups: every coffee; the single-origin coffees from `home_country` and from
    elsewhere, and by processing method (a blend has no one origin or method); and each
    shop. A group of fewer than `min_rows` coffees is left out.
    """
    tasted = flavors.select("coffee_id", "category").unique()
    single = (
        origins.group_by("coffee_id")
        .agg(pl.len().alias("origins"), pl.col("country", "processing_method").first())
        .filter(pl.col("origins") == 1)
    )
    coffees = flavors.select("coffee_id", "shop").unique().join(single, on="coffee_id", how="left")
    method = pl.col("processing_method")

    def dimension(name: str, group: pl.Expr, rows: pl.Expr) -> pl.DataFrame:
        return coffees.filter(rows).select(
            "coffee_id", pl.lit(name).alias("dimension"), group.alias("group")
        )

    home = pl.col("country") == home_country
    everyone = pl.lit(True)
    groups = pl.concat(
        [
            dimension("all", pl.lit("every coffee"), everyone),
            dimension(
                "origin",
                pl.when(home).then(pl.lit(home_country)).otherwise(pl.lit("elsewhere")),
                pl.col("country").is_not_null(),
            ),
            dimension("process", method, method.is_not_null() & (method != UNCLASSIFIED)),
            dimension("shop", pl.col("shop"), everyone),
        ]
    )
    sizes = (
        groups.group_by("dimension", "group")
        .agg(pl.len().alias("coffees"))
        .filter(pl.col("coffees") >= min_rows)
    )
    named = (
        groups.join(tasted, on="coffee_id")
        .group_by("dimension", "group", "category")
        .agg(pl.len().alias("named"))
    )
    return (
        sizes.join(tasted.select("category").unique(), how="cross")
        .join(named, on=["dimension", "group", "category"], how="left")
        .with_columns((100 * pl.col("named").fill_null(0) / pl.col("coffees")).alias("share_pct"))
        .drop("named")
        .sort(
            pl.col("dimension").cast(pl.Enum(["all", "origin", "process", "shop"])),
            pl.col("coffees"),
            "group",
            "category",
            descending=[False, True, False, False],
        )
    )


def flavor_profiles_figure(profiles: pl.DataFrame) -> Figure:
    """One row per group, one column per category, ordered by how often every coffee
    names it: what sets a group apart is where its row departs from the first."""
    everyone = profiles.filter(pl.col("dimension") == "all").sort("share_pct", descending=True)
    categories = everyone["category"].to_list()
    grid = profiles.pivot(
        on="category", index=["dimension", "group", "coffees"], values="share_pct"
    )
    figure, ax = canvas(
        "What the roasters say their coffees taste of",
        "Share of each group's coffees naming a note of the SCA flavour category",
    )
    _share_grid(
        ax,
        grid.select(categories).to_numpy(),
        [category.replace("_", "/\n") for category in categories],  # "nutty/\ncocoa"
        [_group_label(row) for row in grid.iter_rows(named=True)],
    )
    # A gap between the kinds of group: origin, process, shop.
    starts = [
        i for i, kind in enumerate(grid["dimension"]) if i and kind != grid["dimension"][i - 1]
    ]
    for start in starts:
        ax.axhline(start - 0.5, color=SURFACE, linewidth=4)
    ax.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    return figure


def _group_label(row: Mapping[str, object]) -> str:
    """ "from Mexico (23)", "washed process (43)": a process called `other` must not read
    as the flavour category of the same name."""
    group = {"origin": "from {}", "process": "{} process"}.get(str(row["dimension"]), "{}")
    return f"{group.format(row['group'])} ({row['coffees']})"


def flavor_prices(flavors: pl.DataFrame, offers: pl.DataFrame, min_rows: int) -> pl.DataFrame:
    """Whether coffees said to taste of a category cost more: per category, how much
    dearer its coffees are than the other coffees with notes, with an interval.

    Prices are compared within a shop and a bag size - each offer's price per kilogram
    over the median of its shop's offers of that size - because the shop and the size
    set most of a price, and a category common in one shop would otherwise carry that
    shop's prices. A coffee's price is the geometric mean of its offers' ratios, and the
    premium the ratio of the two groups' geometric means, minus one. (Not medians: most
    of a shop's coffees sit exactly at its median, and every difference came out 0%.)

    Nine categories are nine comparisons: at 95% each, one would clear the bar by chance
    about half the time. So the interval is family-wise - 95% for all of them together
    (Bonferroni) - and a category with fewer than `min_rows` coffees on either
    side is not compared. Descriptive, not causal: a category is also a stand-in for the
    origins it is used to describe.
    """
    priced = offers.filter(
        pl.col("price_mxn_per_kg").is_not_null() & ~pl.col("price_outlier").fill_null(False)
    )
    per_kg = pl.col("price_mxn_per_kg")
    relative = (
        priced.with_columns((per_kg / per_kg.median().over("shop", "bag_grams")).log().alias("log"))
        .group_by("coffee_id")
        .agg(pl.col("log").mean())
    )
    tasted = flavors.select("coffee_id", "category").unique()
    described = relative.join(tasted.select("coffee_id").unique(), on="coffee_id")
    sides = {}
    for category in sorted(tasted["category"].unique()):
        named = tasted.filter(pl.col("category") == category).select("coffee_id")
        inside = described.join(named, on="coffee_id")["log"].to_numpy()
        outside = described.join(named, on="coffee_id", how="anti")["log"].to_numpy()
        if min(len(inside), len(outside)) >= min_rows:
            sides[category] = inside, outside
    tail = 0.05 / max(len(sides), 1) / 2
    rows = []
    for number, (category, (inside, outside)) in enumerate(sides.items()):
        seed = PRICE_SEED + 2 * number
        differences = bootstrap_means(inside, PRICE_RESAMPLES, seed) - bootstrap_means(
            outside, PRICE_RESAMPLES, seed + 1
        )
        rows.append(
            {
                "category": category,
                "coffees": len(inside),
                "others": len(outside),
                "premium_pct": 100 * (np.exp(inside.mean() - outside.mean()) - 1),
                "ci_low_pct": 100 * (np.exp(np.quantile(differences, tail)) - 1),
                "ci_high_pct": 100 * (np.exp(np.quantile(differences, 1 - tail)) - 1),
                "probability_dearer": float((differences > 0).mean()),
                "family_level": 1 - 2 * tail,
            }
        )
    return pl.DataFrame(rows, schema=FLAVOR_PRICES).sort("premium_pct", descending=True)


FLAVOR_PRICES = pl.Schema(
    {
        "category": pl.String,
        "coffees": pl.Int64,  # coffees with a note of the category
        "others": pl.Int64,  # coffees with notes, none of the category
        "premium_pct": pl.Float64,
        "ci_low_pct": pl.Float64,
        "ci_high_pct": pl.Float64,
        "probability_dearer": pl.Float64,
        "family_level": pl.Float64,  # the confidence of each interval, after Bonferroni
    }
)
FLAVOR_CLUSTERS = pl.Schema(
    {
        "k": pl.Int64,
        "silhouette": pl.Float64,
        "structure": pl.String,
        "chosen": pl.Boolean,  # the k with the highest silhouette
        "cluster": pl.Int64,  # 1 is the largest
        "coffees": pl.Int64,
        "profile": pl.String,  # the categories at least half its coffees name
    }
)
# Kaufman and Rousseeuw (1990), Finding Groups in Data: what a mean silhouette means.
STRUCTURE = [(0.25, "none"), (0.50, "weak"), (0.70, "reasonable"), (1.0, "strong")]


def flavor_clusters(flavors: pl.DataFrame) -> pl.DataFrame:
    """Whether the coffees fall into flavour types: each coffee as the set of categories
    its notes belong to, clustered with each number of clusters in CLUSTER_COUNTS, and
    how well each clustering holds together (its mean silhouette).

    The silhouette is measured with the Jaccard distance, because the data is presence
    and absence: two coffees that both lack roasted notes are not alike for that. The
    clusters are k-means', seeded: clustering on the Jaccard distance itself was tried
    (2026-09-28) and average linkage peeled the outliers off one or two at a time (92
    coffees and 2 at k=2), complete linkage held together worse (0.22 at best against
    0.31). A cluster is described by the categories at least half of its coffees name,
    and numbered from the largest; `structure` reads the silhouette on Kaufman and
    Rousseeuw's scale, so a table of weak clusters says it is one.
    """
    wide = (
        flavors.select("coffee_id", "category")
        .unique()
        .with_columns(pl.lit(True).alias("named"))
        .pivot(on="category", index="coffee_id", values="named")
        .fill_null(False)
        .sort("coffee_id")
    )
    categories = sorted(column for column in wide.columns if column != "coffee_id")
    presence = wide.select(categories).to_numpy().astype(bool)
    # Never more clusters than there are distinct sets of categories to put in them.
    distinct = len(np.unique(presence, axis=0)) if len(presence) else 0
    counts = [k for k in CLUSTER_COUNTS if k <= distinct and k < len(presence)]
    if not counts:
        return pl.DataFrame(schema=FLAVOR_CLUSTERS)
    distances = pairwise_distances(presence, metric="jaccard")
    runs = []
    for k in counts:
        labels = KMeans(n_clusters=k, n_init=20, random_state=CLUSTER_SEED).fit_predict(
            presence.astype(float)
        )
        runs.append((k, float(silhouette_score(distances, labels, metric="precomputed")), labels))
    best = max(score for _, score, _ in runs)
    rows = []
    for k, score, labels in runs:
        clusters = sorted(
            (_profile(presence[labels == label], categories) for label in range(k)),
            key=lambda cluster: (-cluster[0], cluster[1]),  # the largest first
        )
        rows += [
            {
                "k": k,
                "silhouette": score,
                "structure": next(name for top, name in STRUCTURE if score <= top),
                "chosen": score == best,
                "cluster": number,
                "coffees": size,
                "profile": profile,
            }
            for number, (size, profile) in enumerate(clusters, start=1)
        ]
    return pl.DataFrame(rows, schema=FLAVOR_CLUSTERS)


def _profile(members: np.ndarray, categories: list[str]) -> tuple[int, str]:
    """A cluster's size, and the categories at least half of its coffees name, the most
    named first."""
    shares = members.mean(axis=0)
    held = sorted(
        (-share, name) for share, name in zip(shares, categories, strict=True) if share >= 0.5
    )
    profile = ", ".join(f"{name} {-100 * share:.0f}%" for share, name in held)
    return len(members), profile or "no category in half of them"


def shelf_line() -> pl.Expr:
    """A product and what its presentation declares, as one label: "ground",
    "ground, sweetened", "instant, decaf"."""
    return pl.concat_str(
        pl.col("product"),
        pl.when(pl.col("sweetened")).then(pl.lit(", sweetened")).otherwise(pl.lit("")),
        pl.when(pl.col("decaf")).then(pl.lit(", decaf")).otherwise(pl.lit("")),
    ).alias("line")


def _shelf_summary(prices: pl.DataFrame, *by: str) -> pl.DataFrame:
    """The median price per kilogram, and how many prices and stores it rests on. The
    median, because a promotion or a pharmacy's markup is one shelf, not the market."""
    return prices.group_by(*by).agg(
        pl.col("price_mxn_per_kg").median().alias("median_mxn_per_kg"),
        pl.len().alias("prices"),
        pl.struct("store", "latitude", "longitude").n_unique().alias("stores"),
    )


def consumer_prices_by_fortnight(prices: pl.DataFrame, city: str) -> pl.DataFrame:
    """What a kilogram of each line cost on the shelf, fortnight by fortnight, across
    the country and in the city."""
    lined = prices.with_columns(shelf_line())
    scoped = pl.concat(
        [
            lined.with_columns(pl.lit(NATIONAL).alias("scope")),
            lined.filter(pl.col("state") == city).with_columns(pl.lit(CITY).alias("scope")),
        ]
    )
    return _shelf_summary(scoped, "scope", "line", "fortnight").sort("scope", "line", "fortnight")


def consumer_prices_by_borough(prices: pl.DataFrame) -> pl.DataFrame:
    """Where in the city a kilogram costs what, per borough and line, over every
    fortnight so far. PROFECO visits few stores per borough - `stores` says how few -
    so a borough's figure is its supermarkets' prices, not its residents' spending."""
    in_city = prices.filter(pl.col("borough_id").is_not_null()).with_columns(shelf_line())
    return _shelf_summary(in_city, "borough_id", "borough", "line").sort("borough", "line")


def consumer_prices_by_state(
    prices: pl.DataFrame, production: pl.DataFrame, states: Mapping[str, str]
) -> pl.DataFrame:
    """What shoppers pay for a kilogram of plain ground or instant coffee in each state,
    beside what the state's growers were paid for a kilogram of cherry in SIAP's latest
    year, where it grows coffee. Cherry is not what the shelf sells: it takes several
    kilograms of it to make one of roasted coffee, a factor this table does not assume.
    """
    plain = prices.filter(~pl.col("sweetened") & ~pl.col("decaf"))
    shelves = _shelf_summary(plain, "state", "product")
    spelling = {state: states.get(fold(state)) for state in shelves["state"].unique().to_list()}
    growers = (
        production.filter(pl.col("year") == production["year"].max())
        .group_by(pl.col("state").alias("siap_state"))
        .agg(
            (pl.col("value_mxn").sum() / pl.col("production_t").sum() / 1000).alias(
                "cherry_mxn_per_kg"
            ),
            pl.col("year").first().alias("siap_year"),
        )
    )
    return (
        shelves.with_columns(
            pl.col("state").replace_strict(spelling, return_dtype=pl.String).alias("siap_state")
        )
        .join(growers, on="siap_state", how="left")
        .drop("siap_state")
        .sort("product", "median_mxn_per_kg", descending=[False, True])
    )


# The shelf's steps of the ladder: its line, and the unit its price is per.
SHELF_STEPS = {
    "ground, sweetened": ("supermarket, ground + sugar", "kg of ground coffee and sugar"),
    "ground": ("supermarket, ground", "kg of ground coffee"),
    "instant": ("supermarket, instant", "kg of instant coffee"),
}


def green_coffee_in_pesos(indicators: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """The World Bank's monthly green coffee prices in pesos per kilogram, month by month
    since the peso-dollar series starts (November 1993): each month's price at the mean
    of that month's daily rates, the way FRED averages its own monthly series."""
    monthly_rates = rates.group_by(pl.col("date").dt.truncate("1mo").alias("period")).agg(
        pl.col("mxn_per_usd").mean(), pl.len().alias("rate_days")
    )
    return (
        indicators.filter(pl.col("frequency") == "monthly")
        .join(monthly_rates, on="period", how="inner")
        .select(
            "period",
            "indicator",
            "usd_cents_per_lb",
            "mxn_per_usd",
            "rate_days",
            (
                pl.col("usd_cents_per_lb") / CENTS_PER_LB_PER_USD_PER_KG * pl.col("mxn_per_usd")
            ).alias("mxn_per_kg"),
        )
        .sort("indicator", "period")
    )


def price_ladder(
    prices: pl.DataFrame,
    offers: pl.DataFrame,
    production: pl.DataFrame,
    green: pl.DataFrame,
    city: str,
) -> pl.DataFrame:
    """A kilogram of coffee at each step the data reaches, in pesos: the cherry at the
    farm gate (Mexico, SIAP's latest year), green coffee at the port (the latest month
    of other mild Arabicas, in pesos), the supermarket's shelf and the specialty
    roaster's shop (both in the city). Each step's unit is its own - a kilogram of
    cherry, of green coffee, of instant, of coffee and sugar - and is said beside it:
    the steps are not one product marked up, and no conversion between them is assumed.
    Nor is each step's figure the same statistic, and `measure` says which it is."""
    rows = []
    if not production.is_empty():
        latest = production.filter(pl.col("year") == production["year"].max())
        cherry = latest.select(
            pl.col("value_mxn").sum() / pl.col("production_t").sum() / 1000
        ).item()
        rows.append(
            ("cherry at the farm gate", "SIAP", "kg of coffee cherry", "value over volume",
             cherry, latest.height, str(latest["year"][0]))
        )  # fmt: skip
    milds = green.filter(pl.col("indicator") == GREEN_STEP)
    if not milds.is_empty():
        month = milds.filter(pl.col("period") == milds["period"].max()).row(0, named=True)
        rows.append(
            ("green coffee at the port", "World Bank, FRED", "kg of green coffee",
             "the month's price", month["mxn_per_kg"], 1, month["period"].strftime("%Y-%m"))
        )  # fmt: skip
    shelf = prices.filter(pl.col("state") == city).with_columns(shelf_line())
    for line, (step, unit) in SHELF_STEPS.items():
        on_shelf = shelf.filter(pl.col("line") == line)
        if not on_shelf.is_empty():
            first, last = on_shelf.select(
                pl.col("date").min().alias("first"), pl.col("date").max().alias("last")
            ).row(0)
            period = f"{first} to {last}"
            rows.append(
                (step, "PROFECO", unit, "median", on_shelf["price_mxn_per_kg"].median(),
                 on_shelf.height, period)
            )  # fmt: skip
    bags = offers.filter(
        (pl.col("price_mxn_per_kg") > 0) & ~pl.col("price_outlier").fill_null(False)
    )
    if not bags.is_empty():
        rows.append(
            ("specialty roaster", "roasters", "kg of roasted coffee", "median",
             bags["price_mxn_per_kg"].median(), bags.height, str(bags["observed_on"].max()))
        )  # fmt: skip
    schema = {"step": pl.String, "source": pl.String, "unit": pl.String,
              "measure": pl.String, "mxn_per_kg": pl.Float64, "observations": pl.Int64,
              "period": pl.String}  # fmt: skip
    return pl.DataFrame(rows, schema=schema, orient="row").sort("mxn_per_kg")


def price_ladder_figure(ladder: pl.DataFrame) -> Figure:
    """Pure magnitude, one hue: a kilogram at each step, each labelled with its unit."""
    figure, ax = canvas(
        "A kilogram of coffee, from the farm to the shelf",
        "pesos per kg of what each step sells; shelves and shops in Mexico City",
    )
    ax.barh(ladder["step"], ladder["mxn_per_kg"], color=SERIES[0], height=0.62)
    for step, value, unit in zip(ladder["step"], ladder["mxn_per_kg"], ladder["unit"], strict=True):
        label = f"  ${value:,.0f} per {unit}"
        ax.text(value, step, label, va="center", color=SECONDARY, fontsize=8)
    ax.margins(x=0.75)
    value_grid(ax)
    figure.tight_layout()
    return figure


def consumer_prices_figure(national: pl.DataFrame) -> Figure:
    """Four lines in one unit, one axis, each named at its end: the shelf price of a
    kilogram, fortnight by fortnight, across the country."""
    figure, ax = canvas(
        "Coffee on Mexico's supermarket shelves",
        "median pesos per kilogram, by fortnight (PROFECO)",
    )
    for line, colour in zip(DRAWN_LINES, SERIES, strict=True):
        series = national.filter(pl.col("line") == line).sort("fortnight")
        if series.is_empty():
            continue
        days = series["fortnight"].to_list()
        values = series["median_mxn_per_kg"].to_list()
        ax.plot(days, values, color=colour, linewidth=2, marker="o", markersize=3)
        ax.text(days[-1], values[-1], f"  {line}", color=colour, fontsize=9, va="center")
    # Room on the right for the end labels, none on the left before the first fortnight.
    first, last = national.select(
        pl.col("fortnight").min().alias("first"), pl.col("fortnight").max().alias("last")
    ).row(0)
    ax.set_xlim(first - timedelta(days=6), last + (last - first) * 0.3)
    ax.set_ylim(bottom=0)
    figure.autofmt_xdate()
    value_grid(ax, axis="y")
    figure.tight_layout()
    return figure


def green_coffee_figure(green: pl.DataFrame) -> Figure:
    """Two series in one unit, one axis, each named at its end: green coffee in pesos a
    kilogram, month by month since the peso-dollar series starts."""
    figure, ax = canvas(
        "Green coffee in pesos",
        "pesos per kilogram, monthly: the World Bank's price at the month's peso-dollar rate",
    )
    for (indicator, label), colour in zip(GREEN_LABELS.items(), SERIES, strict=False):
        series = green.filter(pl.col("indicator") == indicator).sort("period")
        if series.is_empty():
            continue
        months = series["period"].to_list()
        values = series["mxn_per_kg"].to_list()
        ax.plot(months, values, color=colour, linewidth=1.6)
        ax.text(months[-1], values[-1], f"  {label}", color=colour, fontsize=9, va="center")
    first, last = green.select(
        pl.col("period").min().alias("first"), pl.col("period").max().alias("last")
    ).row(0)
    ax.set_xlim(first, last + (last - first) * 0.12)
    ax.set_ylim(bottom=0)
    value_grid(ax, axis="y")
    figure.tight_layout()
    return figure
