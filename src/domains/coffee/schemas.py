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

from domains.coffee.config import CleaningConfig

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

# INEGI's establishment register, one row per business. Every field arrives as text.
DENUE_ESTABLISHMENTS = pa.DataFrameSchema(
    name="denue_cafes",
    coerce=True,
    unique=["Id"],
    columns={
        "Id": _text(),
        "Nombre": _text(nullable=True),
        "Clase_actividad": _text(),
        "CLASE_ACTIVIDAD_ID": pa.Column(pl.String, pa.Check.str_matches(r"^\d{6}$")),
        # entity(2) + municipality(3) + locality(4). The first five characters are the
        # borough's CVEGEO, which is what the spatial join is checked against.
        "AreaGeo": pa.Column(pl.String, pa.Check.str_matches(r"^\d{9}$")),
        "Latitud": pa.Column(pl.Float64, pa.Check.in_range(-90, 90)),
        "Longitud": pa.Column(pl.Float64, pa.Check.in_range(-180, 180)),
        # Size band of the workforce ("0 a 5 personas"), the only size DENUE publishes.
        "Estrato": _text(nullable=True),
    },
)

# One row per OSM element, already flattened out of the element/tags shape.
OSM_PLACES = pa.DataFrameSchema(
    name="osm_places",
    coerce=True,
    unique=["type", "id"],
    columns={
        "type": pa.Column(pl.String, pa.Check.isin(["node", "way", "relation"])),
        "id": pa.Column(pl.Int64),
        # Nullable: a crowd-sourced element can arrive without a point, and one bad
        # element must not stop the pipeline. `clean` drops them and says how many.
        "latitude": pa.Column(pl.Float64, pa.Check.in_range(-90, 90), nullable=True),
        "longitude": pa.Column(pl.Float64, pa.Check.in_range(-180, 180), nullable=True),
        "name": _text(nullable=True),
        "brand": _text(nullable=True),
        "amenity": _text(),
        "cuisine": _text(nullable=True),
    },
)

# What `geo.read_areas` produces from a boundary layer, whatever the layer was.
AREAS = pa.DataFrameSchema(
    name="areas",
    strict=True,
    unique=["area_id"],
    columns={
        "area_id": _text(),
        "area_name": _text(),
        "area_km2": pa.Column(pl.Float64, pa.Check.gt(0)),
        # The polygon as WKB in WGS84: readable with any GIS, and with DuckDB's
        # ST_GeomFromWKB, without this project's code.
        "boundary": pa.Column(pl.Binary),
    },
)


def _amount(nullable: bool = False) -> pa.Column:
    return pa.Column(pl.Float64, pa.Check.ge(0), nullable=nullable)


# SIAP's closing statistics: one row per district x CADER x municipality x cycle x
# water regime x crop. No uniqueness is claimed: even that full key repeats twice in the
# 2025 file, and one municipality can sit in several CADERs (Ocosingo is in three).
SIAP_AGRICOLA = pa.DataFrameSchema(
    name="siap_agricola",
    coerce=True,
    columns={
        "Anio": pa.Column(pl.Int64, pa.Check.in_range(2000, 2100)),
        "Idestado": pa.Column(pl.Int64, pa.Check.in_range(1, 32)),
        "Nomestado": _text(),
        "Idmunicipio": pa.Column(pl.Int64, pa.Check.in_range(1, 999)),
        "Nommunicipio": _text(),
        "Nommodalidad": _text(),
        "Idcultivo": _text(),
        "Nomcultivo": _text(),
        "Nomunidad": _text(),
        "Sembrada": _amount(),  # hectares
        "Cosechada": _amount(),
        "Siniestrada": _amount(),
        "Volumenproduccion": _amount(),
        # Empty where nothing was harvested: a yield or a price of nothing is undefined.
        "Rendimiento": _amount(nullable=True),
        "Preciomediorural": _amount(nullable=True),  # MXN per unit
        "Valorproduccion": _amount(),  # MXN
    },
)

RAW_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "cqi_2018": CQI_2018,
    "cqi_2023": CQI_2023,
    "psd_coffee": PSD_COFFEE,
    "cdmx_boroughs": AREAS,
    "denue_cafes": DENUE_ESTABLISHMENTS,
    "osm_places": OSM_PLACES,
    # The API is held to the file's contract: one table, two ways to reach it.
    "fas_psd_coffee": PSD_COFFEE,
    "siap_agricola": SIAP_AGRICOLA,
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


BOROUGHS = pa.DataFrameSchema(
    name="boroughs",
    strict=True,
    unique=["borough_id"],
    columns={
        "borough_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
        "borough": pa.Column(pl.String),
        "area_km2": pa.Column(pl.Float64, pa.Check.gt(0)),
        "boundary": pa.Column(pl.Binary),
    },
)


def coffee_shops_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of `coffee_shops`; the kinds a place can have come from the config."""
    return COFFEE_SHOPS.add_columns(
        {
            "kind": pa.Column(pl.String, pa.Check.isin(rules.kinds)),
            # How the kind was decided: DENUE's name read by the rules, or OSM's own tag.
            "kind_basis": pa.Column(pl.String, pa.Check.isin(["name", "tag"])),
            # The same place in the other register, when both list it.
            # Nulls do not collide: only the links themselves must be one-to-one.
            "matched_shop_id": pa.Column(
                pl.String,
                pa.Check(
                    lambda data: data.lazyframe.select(
                        ~pl.col(data.key).drop_nulls().is_duplicated().any()
                    ),
                    error="a place is linked to one place at most",
                ),
                nullable=True,
            ),
        }
    )


# The columns every place has, before its kind is read. `coffee_shops_schema` completes it.
COFFEE_SHOPS = pa.DataFrameSchema(
    name="coffee_shops",
    strict=True,
    unique=["shop_id"],
    columns={
        # "<source>-<the source's own id>": stable across runs, and it says where the
        # row came from without reading another column.
        "shop_id": pa.Column(pl.String),
        "source": pa.Column(pl.String, pa.Check.isin(["denue", "osm"])),
        "name": pa.Column(pl.String, nullable=True),
        "brand": pa.Column(pl.String, nullable=True),
        "employees_band": pa.Column(pl.String, nullable=True),
        "latitude": pa.Column(pl.Float64, pa.Check.in_range(-90, 90)),
        "longitude": pa.Column(pl.Float64, pa.Check.in_range(-180, 180)),
        # Null when the point lands outside every borough: a fact worth keeping, not a
        # reason to drop the shop.
        "borough_id": pa.Column(pl.String, nullable=True),
        "borough": pa.Column(pl.String, nullable=True),
        # What the source itself says the borough is. DENUE carries one, OSM does not,
        # so this is the column the spatial join is audited against.
        "declared_borough_id": pa.Column(pl.String, nullable=True),
    },
)


def clean_schemas(rules: CleaningConfig) -> dict[str, pa.DataFrameSchema]:
    """One strict contract per clean table: the domain's promise to every reader."""
    return {
        "coffee_reviews": coffee_reviews_schema(rules),
        "market_context": MARKET_CONTEXT,
        "boroughs": BOROUGHS,
        "coffee_shops": coffee_shops_schema(rules),
        "mexico_production": MEXICO_PRODUCTION,
    }


MEXICO_PRODUCTION = pa.DataFrameSchema(
    name="mexico_production",
    strict=True,
    unique=["municipality_id", "year"],
    columns={
        "year": pa.Column(pl.Int64),
        "state_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2}$")),
        "state": pa.Column(pl.String),
        # INEGI's CVEGEO (state + municipality), the key the boundary layers use.
        "municipality_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
        "municipality": pa.Column(pl.String),
        "planted_ha": pa.Column(pl.Float64, pa.Check.ge(0)),
        "harvested_ha": pa.Column(pl.Float64, pa.Check.ge(0)),
        "lost_ha": pa.Column(pl.Float64, pa.Check.ge(0)),
        "production_t": pa.Column(pl.Float64, pa.Check.ge(0)),  # tonnes of cherry
        "value_mxn": pa.Column(pl.Float64, pa.Check.ge(0)),
        # Derived from the totals, so they stay right after summing CADERs; null where
        # the denominator is zero.
        "yield_t_per_ha": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        "rural_price_mxn_per_t": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    },
)
