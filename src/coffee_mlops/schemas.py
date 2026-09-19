"""Pandera contracts for each raw source.

Raw schemas describe what we *depend on* from upstream: required columns, types
and the unit assumptions the cleaning code relies on. Extra upstream columns are
allowed (`strict=False`). Known bad values that are still well-formed (an
altitude of 190 km, a 0-point cup) pass here and are handled in `clean`.

The 2018 scrape was written by R: missing numbers are "NA" (read as null via the
source config) and missing text is a quoted empty string (normalized in `clean`).
"""

import pandera.polars as pa
import polars as pl

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
# The 2018 scrape names two of them differently.
SENSORY_SCORES_2018 = [
    {"Clean Cup": "Clean.Cup", "Overall": "Cupper.Points"}.get(c, c) for c in SENSORY_SCORES
]

PSD_ATTRIBUTES = [
    "Arabica Production",
    "Bean Exports",
    "Bean Imports",
    "Beginning Stocks",
    "Domestic Consumption",
    "Ending Stocks",
    "Exports",
    "Imports",
    "Other Production",
    "Production",
    "Roast & Ground Exports",
    "Roast & Ground Imports",
    "Robusta Production",
    "Rst,Ground Dom. Consum",
    "Soluble Dom. Cons.",
    "Soluble Exports",
    "Soluble Imports",
    "Total Distribution",
    "Total Supply",
]


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
        "Attribute_Description": pa.Column(pl.String, pa.Check.isin(PSD_ATTRIBUTES)),
        "Unit_Description": pa.Column(pl.String, pa.Check.eq("(1000 60 KG BAGS)")),
        "Value": pa.Column(pl.Float64, pa.Check.ge(0)),
    },
)

RAW_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "cqi_2018": CQI_2018,
    "cqi_2023": CQI_2023,
    "psd_coffee": PSD_COFFEE,
}
