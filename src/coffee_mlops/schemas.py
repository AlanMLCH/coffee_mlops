"""Pandera contracts: one per raw source, and one per clean table.

Raw schemas describe what we *depend on* from upstream: required columns, types
and the unit assumptions the cleaning code relies on. Extra upstream columns are
allowed (`strict=False`). Known bad values that are still well-formed (an
altitude of 190 km, a 0-point cup) pass here and are handled in `clean`.

The 2018 scrape was written by R: missing numbers are "NA" (read as null via the
source config) and missing text is a quoted empty string (normalized in `clean`).

Clean schemas are the opposite: `strict=True`, no coercion. They are our own
contract with every downstream consumer (features, the agent's SQL, the API).
"""

import pandera.polars as pa
import polars as pl

from coffee_mlops.config import CleaningConfig

SENSORY_SCORES = [
    "Aroma",
    "Flavor",
    "Aftertaste",
    "Acidity",
    "Body",
    "Balance",
    "Uniformity",
    "Clean Cup",
    "Sweetness",
    "Overall",
]
# Canonical names in the clean layer.
SENSORY_COLUMNS = [c.lower().replace(" ", "_") for c in SENSORY_SCORES]
# The 2018 scrape names two of them differently.
SENSORY_SCORES_2018 = [
    {"Clean Cup": "Clean.Cup", "Overall": "Cupper.Points"}.get(c, c) for c in SENSORY_SCORES
]

# Upstream attribute -> column in the clean `market_context` table.
PSD_ATTRIBUTES = {
    "Arabica Production": "arabica_production",
    "Bean Exports": "bean_exports",
    "Bean Imports": "bean_imports",
    "Beginning Stocks": "beginning_stocks",
    "Domestic Consumption": "domestic_consumption",
    "Ending Stocks": "ending_stocks",
    "Exports": "exports",
    "Imports": "imports",
    "Other Production": "other_production",
    "Production": "production",
    "Roast & Ground Exports": "roast_ground_exports",
    "Roast & Ground Imports": "roast_ground_imports",
    "Robusta Production": "robusta_production",
    "Rst,Ground Dom. Consum": "roast_ground_domestic_consumption",
    "Soluble Dom. Cons.": "soluble_domestic_consumption",
    "Soluble Exports": "soluble_exports",
    "Soluble Imports": "soluble_imports",
    "Total Distribution": "total_distribution",
    "Total Supply": "total_supply",
}


def _text(nullable: bool = False) -> pa.Column:
    return pa.Column(pl.String, nullable=nullable)


def _count(nullable: bool = False) -> pa.Column:
    return pa.Column(pl.Int64, pa.Check.ge(0), nullable=nullable)


def _score(max_value: float) -> pa.Column:
    return pa.Column(pl.Float64, pa.Check.in_range(0, max_value))


CQI_2018 = pa.DataFrameSchema(
    name="cqi_2018",
    coerce=True,
    columns={
        "Species": pa.Column(pl.String, pa.Check.eq("Arabica")),
        "Country.of.Origin": _text(nullable=True),
        "Region": _text(nullable=True),
        "Variety": _text(nullable=True),
        "Processing.Method": _text(nullable=True),
        "Color": _text(nullable=True),
        "Harvest.Year": _text(nullable=True),
        "Grading.Date": _text(),
        "altitude_mean_meters": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        # Moisture is a fraction here (0.12); the 2023 scrape uses a percentage.
        "Moisture": pa.Column(pl.Float64, pa.Check.in_range(0, 1)),
        "Category.One.Defects": _count(),
        "Category.Two.Defects": _count(),
        "Quakers": _count(nullable=True),
        **{c: _score(10) for c in SENSORY_SCORES_2018},
        "Total.Cup.Points": _score(100),
    },
)

CQI_2023 = pa.DataFrameSchema(
    name="cqi_2023",
    coerce=True,
    columns={
        "Country of Origin": _text(),
        "Region": _text(nullable=True),
        "Variety": _text(nullable=True),
        "Processing Method": _text(nullable=True),
        "Color": _text(nullable=True),
        "Harvest Year": _text(nullable=True),
        "Grading Date": _text(),
        # Free text such as "1700-1930" or "1200"; parsed in clean.
        "Altitude": _text(nullable=True),
        "Moisture Percentage": pa.Column(pl.Float64, pa.Check.in_range(0, 100)),
        "Category One Defects": _count(),
        "Category Two Defects": _count(),
        "Quakers": _count(),
        **{c: _score(10) for c in SENSORY_SCORES},
        "Defects": pa.Column(pl.Float64, pa.Check.ge(0)),
        "Total Cup Points": _score(100),
    },
)

PSD_COFFEE = pa.DataFrameSchema(
    name="psd_coffee",
    coerce=True,
    unique=["Country_Code", "Market_Year", "Attribute_ID"],
    columns={
        "Commodity_Description": pa.Column(pl.String, pa.Check.eq("Coffee, Green")),
        "Country_Code": _text(),
        "Country_Name": _text(),
        "Market_Year": pa.Column(pl.Int64, pa.Check.in_range(1960, 2100)),
        "Attribute_ID": pa.Column(pl.Int64),
        # A new attribute changes the pivot in clean: fail and decide, don't guess.
        "Attribute_Description": pa.Column(pl.String, pa.Check.isin(list(PSD_ATTRIBUTES))),
        "Unit_Description": pa.Column(pl.String, pa.Check.eq("(1000 60 KG BAGS)")),
        "Value": pa.Column(pl.Float64, pa.Check.ge(0)),
    },
)

RAW_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "cqi_2018": CQI_2018,
    "cqi_2023": CQI_2023,
    "psd_coffee": PSD_COFFEE,
}


def coffee_reviews_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of the clean `coffee_reviews` table; vocabularies and ranges come from config."""
    altitude_low, altitude_high = rules.altitude_m
    return pa.DataFrameSchema(
        name="coffee_reviews",
        strict=True,
        unique=["review_id"],
        columns={
            "review_id": pa.Column(pl.String),
            "snapshot": pa.Column(pl.String, pa.Check.isin(["cqi_2018", "cqi_2023"])),
            "country": pa.Column(pl.String),
            "region": pa.Column(pl.String, nullable=True),
            "variety": pa.Column(pl.String, nullable=True),
            "processing_method": pa.Column(
                pl.String, pa.Check.isin(set(rules.processing_methods.values())), nullable=True
            ),
            "color": pa.Column(
                pl.String, pa.Check.isin({c for c in rules.colors.values() if c}), nullable=True
            ),
            "grading_date": pa.Column(pl.Date),
            "altitude_m": pa.Column(
                pl.Float64, pa.Check.in_range(altitude_low, altitude_high), nullable=True
            ),
            "moisture_pct": pa.Column(pl.Float64, pa.Check.in_range(0, 100), nullable=True),
            "category_one_defects": pa.Column(pl.Int64, pa.Check.ge(0)),
            "category_two_defects": pa.Column(pl.Int64, pa.Check.ge(0)),
            "quakers": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
            **{c: pa.Column(pl.Float64, pa.Check.in_range(0, 10)) for c in SENSORY_COLUMNS},
            "total_cup_points": pa.Column(pl.Float64, [pa.Check.gt(0), pa.Check.le(100)]),
        },
    )


MARKET_CONTEXT = pa.DataFrameSchema(
    name="market_context",
    strict=True,
    unique=["country", "market_year"],
    columns={
        "country": pa.Column(pl.String),
        "market_year": pa.Column(pl.Int64),
        # Thousands of 60 kg bags. Null means "not reported", never zero.
        **{
            c: pa.Column(pl.Float64, pa.Check.ge(0), nullable=True) for c in PSD_ATTRIBUTES.values()
        },
    },
)
