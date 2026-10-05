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

from domains.coffee.config import (
    UNCLASSIFIED,
    BoroughProfileConfig,
    CleaningConfig,
    HouseholdSpendingConfig,
)

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

# The ICO's indicator prices, in the order its page prints them. Here, not beside the
# reader: the prediction API imports these contracts, and the sources package needs httpx.
ICO_INDICATORS = ("i_cip", "colombian_milds", "other_milds", "brazilian_naturals", "robustas")

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
        # The register's edition the place entered it, as a month: "2024-11" (twice in
        # 9,860 with a space, "2013 07").
        "Fecha_Alta": pa.Column(
            pl.String, pa.Check.str_matches(r"^\d{4}[- ]\d{2}$"), nullable=True
        ),
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

# One row per offer - a roaster's product in one size - as the shops listed it.
ROASTER_CATALOGS = pa.DataFrameSchema(
    name="roaster_catalogs",
    coerce=True,
    unique=["shop", "variant_id"],
    columns={
        "shop": _text(),
        "platform": pa.Column(pl.String, pa.Check.isin(["shopify", "squarespace"])),
        "product_id": _text(),
        "variant_id": _text(),
        "title": _text(),
        "variant_title": _text(nullable=True),
        "price": pa.Column(pl.Float64, pa.Check.ge(0)),  # MXN, as listed
        "platform_grams": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),  # 0 = not set
        "url": pa.Column(pl.String, pa.Check.str_startswith("https://")),
        "tags": _text(nullable=True),
        "body_html": _text(nullable=True),
        "page_html": _text(nullable=True),  # only for shops whose attributes live there
    },
)


def _price() -> pa.Column:
    return pa.Column(pl.Float64, pa.Check.gt(0))


# The Pink Sheet's monthly prices: only the months and the two coffee columns are
# depended on; its other seventy commodities pass through unread.
WORLD_BANK_PRICES = pa.DataFrameSchema(
    name="world_bank_prices",
    coerce=True,
    unique=["column_1"],
    columns={
        "column_1": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}M(0[1-9]|1[0-2])$")),
        "Coffee, Arabica": _price(),  # $/kg
        "Coffee, Robusta": _price(),
    },
)

# One download of the ICO's page: the days of one month, each once.
ICO_PRICES = pa.DataFrameSchema(
    name="ico_prices",
    coerce=True,
    unique=["date"],
    columns={
        "date": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        **{indicator: _price() for indicator in ICO_INDICATORS},  # US cents/lb
    },
)

# PROFECO's shelf prices, coffee only, as its reader leaves them: what the clean layer
# depends on, typed. A presentation that states no size could never become a price
# per kilogram, so it stops the pipeline here.
PROFECO_PRICES = pa.DataFrameSchema(
    name="profeco_prices",
    coerce=True,
    columns={
        "producto": pa.Column(pl.String),
        "presentacion": pa.Column(pl.String, pa.Check.str_matches(r"(?i).*\d\s*(gr|g|kg)\b")),
        "marca": pa.Column(pl.String),
        "precio": _price(),  # pesos, per jar, bag or sachet
        "fecha_registro": pa.Column(
            pl.String, pa.Check.str_matches(r"^(\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4})$")
        ),
        "cadena_comercial": pa.Column(pl.String),
        "giro": pa.Column(pl.String),
        "nombre_comercial": pa.Column(pl.String),
        "estado": pa.Column(pl.String),
        "municipio": pa.Column(pl.String),
        # Inside Mexico's bounding box: a swapped pair or a zero would land outside it.
        "latitud": pa.Column(pl.Float64, pa.Check.in_range(14.0, 33.0)),
        "longitud": pa.Column(pl.Float64, pa.Check.in_range(-119.0, -86.0)),
        "file": pa.Column(pl.String),
    },
)

# FRED's daily peso-dollar rate as downloaded: one business day a row, empty on the days
# no rate was set (US holidays).
FRED_USD_MXN = pa.DataFrameSchema(
    name="fred_usd_mxn",
    coerce=True,
    unique=["observation_date"],
    columns={
        "observation_date": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "DEXMXUS": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),  # pesos per dollar
    },
)

# INEGI withholds a small locality's figures to protect its people, and writes an asterisk.
_CENSUS_FIGURE = pa.Column(pl.String, pa.Check.str_matches(r"^(\d+(\.\d+)?|\*)$"))

# The 2020 Census, principal results by locality (ITER): only the columns read. A total
# row per alcaldia (LOC 0000) and one for the state (MUN 000).
CENSUS_2020 = pa.DataFrameSchema(
    name="census_2020",
    coerce=True,
    unique=["ENTIDAD", "MUN", "LOC"],
    columns={
        "ENTIDAD": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2}$")),
        "MUN": pa.Column(pl.String, pa.Check.str_matches(r"^\d{3}$")),
        "NOM_MUN": _text(),
        "LOC": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}$")),
        "POBTOT": pa.Column(pl.Int64, pa.Check.ge(0)),  # never withheld
        "P_18YMAS": _CENSUS_FIGURE,
        "TVIVHAB": _CENSUS_FIGURE,
        "GRAPROES": _CENSUS_FIGURE,
        "PEA": _CENSUS_FIGURE,
    },
)

# The intercensal survey's estimators, as its file names them, and the clean table's name
# for each. The standard error is left out: the interval and the coefficient say it.
SURVEY_ESTIMATORS = {
    "Valor": "value",
    "Límite inferior de confianza": "ci_low",
    "Límite superior de confianza": "ci_high",
    "Coeficiente de variación": "cv",
}
SURVEY_STANDARD_ERROR = "Error estándar"


def intercensal_schema(profile: BoroughProfileConfig) -> pa.DataFrameSchema:
    """The survey's principal results as downloaded: a row per area and estimator, and
    only the columns the profile reads. "MI" (a sample too small) and "NA" (does not
    apply) arrive as nulls, as the source config says."""
    return pa.DataFrameSchema(
        name=profile.source,
        coerce=True,
        unique=["CVEGEO", "ESTIMADOR"],
        columns={
            "CVEGEO": pa.Column(pl.String, pa.Check.str_matches(r"^\d{9}$")),
            "CVE_ENT": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2}$")),
            "CVE_MUN": pa.Column(pl.String, pa.Check.str_matches(r"^\d{3}$")),
            "NOM_MUN": _text(),
            "CVE_LOC": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}$")),
            "ESTIMADOR": pa.Column(
                pl.String, pa.Check.isin([*SURVEY_ESTIMATORS, SURVEY_STANDARD_ERROR])
            ),
            **{column: pa.Column(pl.Float64, nullable=True) for column in profile.indicators},
        },
    )


# DENUE's count of every activity, per area and staff-size stratum: a row per activity
# code at every level of SCIAN (sector 2 digits, down to class 6), as the service answers.
DENUE_WORKPLACES = pa.DataFrameSchema(
    name="denue_workplaces",
    coerce=True,
    unique=["area", "stratum", "activity"],
    columns={
        "area": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
        "stratum": pa.Column(pl.Int64, pa.Check.in_range(1, 7)),
        "activity": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2,6}$")),
        "establishments": pa.Column(pl.Int64, pa.Check.ge(0)),
    },
)

# FAOSTAT's producer prices of one item, as its bulk file writes them: a row per country,
# year, period (the year's value or a month's) and element. Flags from the file's own
# legend: A official, B a break in the series, E estimated, I imputed, X from another
# organisation.
FAOSTAT_PRICES = pa.DataFrameSchema(
    name="faostat_prices",
    coerce=True,
    unique=["Area Code", "Element Code", "Year", "Months"],
    columns={
        "Area Code": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
        "Area": _text(),
        "Item Code": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
        "Item": _text(),
        "Element Code": pa.Column(pl.String, pa.Check.isin(["5530", "5531", "5532", "5539"])),
        "Element": _text(),
        "Year": pa.Column(pl.Int64, pa.Check.in_range(1960, 2100)),
        "Months": _text(),
        "Unit": pa.Column(pl.String, nullable=True),  # the index has none
        "Value": pa.Column(pl.Float64, pa.Check.ge(0)),
        "Flag": pa.Column(pl.String, pa.Check.isin(["A", "B", "E", "I", "X"])),
    },
)

_DAY = pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$"))

# The Metro's entries as published: a row per station-line and day. Not unique - one
# December has a station named twice a day - which `clean` resolves and says.
METRO_RIDERSHIP = pa.DataFrameSchema(
    name="metro_ridership",
    coerce=True,
    columns={
        "fecha": _DAY,
        "linea": _text(),
        "estacion": _text(),
        "afluencia": pa.Column(pl.Int64, pa.Check.ge(0)),  # zero: the station was closed
    },
)

# The Metrobús' entries as published: a row per line and day, null before it opened.
METROBUS_RIDERSHIP = pa.DataFrameSchema(
    name="metrobus_ridership",
    coerce=True,
    unique=["fecha", "linea"],
    columns={
        "fecha": _DAY,
        "linea": _text(),
        "afluencia": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    },
)

# The city's GTFS stops, every system's: where each stop is.
GTFS_STOPS = pa.DataFrameSchema(
    name="transit_stops",
    coerce=True,
    unique=["stop_id"],
    columns={
        "stop_id": _text(),
        "stop_name": _text(),
        # Inside Mexico's bounding box: a swapped pair or a zero would land outside it.
        "stop_lat": pa.Column(pl.Float64, pa.Check.in_range(14.0, 33.0)),
        "stop_lon": pa.Column(pl.Float64, pa.Check.in_range(-119.0, -86.0)),
    },
)

# The 2020 Census' total row per urban AGEB, as the domain's reader keeps it: the key and
# the figures used, a withheld one null.
CENSUS_ZONES_RAW = pa.DataFrameSchema(
    name="census_2020_ageb",
    coerce=True,
    unique=["ENTIDAD", "MUN", "LOC", "AGEB"],
    columns={
        "ENTIDAD": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2}$")),
        "MUN": pa.Column(pl.String, pa.Check.str_matches(r"^\d{3}$")),
        "LOC": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}$")),
        "AGEB": pa.Column(pl.String, pa.Check.str_matches(r"^[0-9A-Z]{4}$")),  # "122A"
        "POBTOT": pa.Column(pl.Int64, pa.Check.ge(0)),  # never withheld
        **{
            column: pa.Column(pl.Int64, pa.Check.ge(0), nullable=True)
            for column in ("TVIVHAB", "PEA", "POB65_MAS", "VPH_INTER", "VPH_AUTOM", "VPH_PC")
        },
        "GRAPROES": pa.Column(pl.Float64, pa.Check.in_range(0, 25), nullable=True),
    },
)

# ENIGH's purchases of coffee, as the domain's reader keeps them: text keys with their
# leading zeros. A purchase holds what was paid or, given or harvested, what it was worth.
ENIGH_SPENDING_RAW = pa.DataFrameSchema(
    name="enigh_spending",
    coerce=True,
    columns={
        "folioviv": pa.Column(pl.String, pa.Check.str_matches(r"^\d{10}$")),
        "foliohog": pa.Column(pl.String, pa.Check.str_matches(r"^\d$")),
        "clave": pa.Column(pl.String, pa.Check.str_matches(r"^\d{6}$")),
        "tipo_gasto": pa.Column(pl.String, pa.Check.isin(["G1", "G2", "G3", "G5", "G6", "G7"])),
        "gasto_tri": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        "gas_nm_tri": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        "entidad": pa.Column(pl.String, pa.Check.str_matches(r"^\d{2}$")),
    },
)

# ENIGH's households: a row each, its weight the households it stands for.
ENIGH_HOUSEHOLDS_RAW = pa.DataFrameSchema(
    name="enigh_households",
    coerce=True,
    unique=["folioviv", "foliohog"],
    columns={
        "folioviv": pa.Column(pl.String, pa.Check.str_matches(r"^\d{10}$")),
        "foliohog": pa.Column(pl.String, pa.Check.str_matches(r"^\d$")),
        "ubica_geo": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),  # state + municipality
        "est_dis": pa.Column(pl.String, pa.Check.str_matches(r"^\d{3}$")),  # the design's stratum
        "upm": pa.Column(pl.String, pa.Check.str_matches(r"^\d{7}$")),  # its sampling unit
        "factor": pa.Column(pl.Int64, pa.Check.gt(0)),
        "tot_integ": pa.Column(pl.Int64, pa.Check.ge(1)),
        "ing_cor": pa.Column(pl.Float64, pa.Check.ge(0)),
    },
)

COE_TEXT = ("rank", "score", "farmer", "region", "variety", "process", "weight_kg",
            "weight_lb", "bid", "total", "buyers")  # fmt: skip
# Cup of Excellence Mexico, as the domain's reader keeps a page: a row of a table of lots
# or of sales, every value as the page writes it. A row always names its farm.
CUP_OF_EXCELLENCE_RAW = pa.DataFrameSchema(
    name="cup_of_excellence",
    coerce=True,
    columns={
        "year": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}$")),
        "table": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
        "farm": pa.Column(pl.String),
        **{column: pa.Column(pl.String, nullable=True) for column in COE_TEXT},
    },
)

# INEGI's consumer price index as the API gives it: a month ("2026/08") and its value.
INPC_RAW = pa.DataFrameSchema(
    name="inpc",
    coerce=True,
    unique=["period"],
    columns={
        "period": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}/(0[1-9]|1[0-2])$")),
        "value": pa.Column(pl.Float64, pa.Check.gt(0)),
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
    "roaster_catalogs": ROASTER_CATALOGS,
    "world_bank_prices": WORLD_BANK_PRICES,
    "ico_prices": ICO_PRICES,
    "profeco_prices": PROFECO_PRICES,
    "fred_usd_mxn": FRED_USD_MXN,
    "inpc": INPC_RAW,
    "census_2020": CENSUS_2020,
    "denue_workplaces": DENUE_WORKPLACES,
    "faostat_prices": FAOSTAT_PRICES,
    "metro_ridership": METRO_RIDERSHIP,
    "metrobus_ridership": METROBUS_RIDERSHIP,
    "transit_stops": GTFS_STOPS,
    "census_2020_ageb": CENSUS_ZONES_RAW,
    "cup_of_excellence": CUP_OF_EXCELLENCE_RAW,
    "cdmx_ageb": AREAS,
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
        # The 2020 Census; null until it is downloaded.
        "population": pa.Column(pl.Int64, pa.Check.gt(0), nullable=True),
        "adults": pa.Column(pl.Int64, pa.Check.gt(0), nullable=True),
        "households": pa.Column(pl.Int64, pa.Check.gt(0), nullable=True),
        "schooling_years": pa.Column(pl.Float64, pa.Check.in_range(0, 25), nullable=True),
        "economically_active": pa.Column(pl.Int64, pa.Check.gt(0), nullable=True),
        # DENUE's workplaces of every activity (null without a token), and the jobs they
        # hold estimated from their staff-size bands: each band's midpoint, and 251 for
        # "251 or more" - a floor for the largest.
        "workplaces": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "jobs_estimate": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    },
)


def household_coffee_schema(survey: HouseholdSpendingConfig) -> pa.DataFrameSchema:
    """Contract of `household_coffee`: a household of the survey a row, who it is and what
    it spent on coffee in the quarter - a column for each of the domain's coffees."""
    money = pa.Column(pl.Float64, pa.Check.ge(0))
    return pa.DataFrameSchema(
        name="household_coffee",
        strict=True,
        unique=["year", "household_id"],
        columns={
            "year": pa.Column(pl.Int64, pa.Check.eq(survey.year)),
            "household_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{10}-\d$")),
            "state_id": pa.Column(pl.String, pa.Check.isin(list(survey.states))),
            "state": pa.Column(pl.String, pa.Check.isin(list(survey.states.values()))),
            "municipality_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
            "stratum": pa.Column(pl.String),
            "psu": pa.Column(pl.String),
            "weight": pa.Column(pl.Int64, pa.Check.gt(0)),
            "members": pa.Column(pl.Int64, pa.Check.ge(1)),
            "income_quarter_mxn": money,
            "income_decile": pa.Column(pl.Int64, pa.Check.in_range(1, 10)),
            **{f"{name}_quarter_mxn": money for name in survey.products.values()},
            "own_harvest_quarter_mxn": money,
        },
    )


def borough_profile_schema(profile: BoroughProfileConfig) -> pa.DataFrameSchema:
    """Contract of `borough_profile`: a row per borough and indicator of the survey, the
    value inside its own interval. The indicators a row can name come from the config."""
    return pa.DataFrameSchema(
        name="borough_profile",
        strict=True,
        unique=["borough_id", "indicator"],
        columns={
            "borough_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
            "borough": pa.Column(pl.String),
            "year": pa.Column(pl.Int64, pa.Check.eq(profile.year)),
            "indicator": pa.Column(
                pl.String, pa.Check.isin([i.name for i in profile.indicators.values()])
            ),
            "unit": pa.Column(pl.String),
            # Null where the survey's sample was too small or the figure does not apply.
            "value": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
            "ci_low": pa.Column(pl.Float64, nullable=True),
            "ci_high": pa.Column(pl.Float64, nullable=True),
            "cv": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),  # percent
        },
        checks=[
            pa.Check(
                lambda data: data.lazyframe.select(
                    (pl.col("ci_low") <= pl.col("value"))
                    .and_(pl.col("value") <= pl.col("ci_high"))
                    .fill_null(True)
                ),
                error="an estimate lies inside its own interval",
            )
        ],
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
        # The urban AGEB it falls in (the census' finer zone); null outside every one.
        "zone_id": pa.Column(pl.String, nullable=True),
        # The month of the DENUE edition the place entered the register; OSM keeps none.
        # An entry, not an opening: each economic census adds thousands at once.
        "listed_since": pa.Column(pl.Date, nullable=True),
    },
)


def coffee_shop_history_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of `coffee_shop_history`: every place in every read of its register."""
    return pa.DataFrameSchema(
        name="coffee_shop_history",
        strict=True,
        unique=["shop_id", "snapshot"],
        columns={
            "shop_id": pa.Column(pl.String),
            "source": pa.Column(pl.String, pa.Check.isin(["denue", "osm"])),
            # The day of the read, as the roasters' history writes it.
            "snapshot": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
            "name": pa.Column(pl.String, nullable=True),
            "kind": pa.Column(pl.String, pa.Check.isin(rules.kinds)),
            "latitude": pa.Column(pl.Float64, pa.Check.in_range(-90, 90)),
            "longitude": pa.Column(pl.Float64, pa.Check.in_range(-180, 180)),
            "borough_id": pa.Column(pl.String, nullable=True),
            "borough": pa.Column(pl.String, nullable=True),
        },
    )


def clean_schemas(rules: CleaningConfig) -> dict[str, pa.DataFrameSchema]:
    """One strict contract per clean table: the domain's promise to every reader. The
    survey's profile has its own (`borough_profile_schema`), shaped by its config."""
    return {
        "coffee_reviews": coffee_reviews_schema(rules),
        "market_context": MARKET_CONTEXT,
        "boroughs": BOROUGHS,
        "coffee_shops": coffee_shops_schema(rules),
        "coffee_shop_history": coffee_shop_history_schema(rules),
        "mexico_production": MEXICO_PRODUCTION,
        "producer_prices": PRODUCER_PRICES,
        "roaster_coffees": ROASTER_COFFEES,
        "roaster_origins": roaster_origins_schema(rules),
        "roaster_offers": ROASTER_OFFERS,
        "roaster_offer_history": ROASTER_OFFER_HISTORY,
        "roaster_origin_history": roaster_origin_history_schema(rules),
        "roaster_flavors": roaster_flavors_schema(rules),
        "price_indicators": PRICE_INDICATORS,
        "consumer_prices": CONSUMER_PRICES,
        "exchange_rates": EXCHANGE_RATES,
        "transit_stations": TRANSIT_STATIONS,
        "census_zones": CENSUS_ZONES,
        "transit_ridership": TRANSIT_RIDERSHIP,
        "cup_of_excellence": cup_of_excellence_schema(rules),
        "consumer_price_index": CONSUMER_PRICE_INDEX,
    }


ROASTER_COFFEES = pa.DataFrameSchema(
    name="roaster_coffees",
    strict=True,
    unique=["shop", "product_id"],
    columns={
        "coffee_id": pa.Column(pl.String, unique=True),  # "<shop>-<product id>"
        "shop": pa.Column(pl.String),
        "product_id": pa.Column(pl.String),
        "title": pa.Column(pl.String),
        "url": pa.Column(pl.String, pa.Check.str_startswith("https://")),
        "description": pa.Column(pl.String, nullable=True),
        "origins": pa.Column(pl.Int64, pa.Check.ge(0)),
    },
)


def cup_of_excellence_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of `cup_of_excellence`: a lot of one year's competition and its sale; its
    state and process in the vocabularies the roasters' sheets use."""
    sheets = rules.roaster_sheets
    positive = pa.Column(pl.Float64, pa.Check.gt(0), nullable=True)
    return pa.DataFrameSchema(
        name="cup_of_excellence",
        strict=True,
        unique=["lot_id"],
        columns={
            "year": pa.Column(pl.Int64, pa.Check.in_range(2000, 2100)),
            "lot_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{3}$")),
            "rank": pa.Column(pl.String, nullable=True),
            "national_winner": pa.Column(pl.Boolean),
            "score": pa.Column(pl.Float64, pa.Check.in_range(80, 100)),
            "farm": pa.Column(pl.String),
            "farmer": pa.Column(pl.String, nullable=True),
            "region": pa.Column(pl.String, nullable=True),
            "state": pa.Column(
                pl.String, pa.Check.isin(sorted(set(sheets.states.values()))), nullable=True
            ),
            "varieties": pa.Column(pl.List(pl.String), nullable=True),
            "processing_method": pa.Column(
                pl.String,
                pa.Check.isin(sorted({*rules.processing_methods.values(), UNCLASSIFIED})),
                nullable=True,
            ),
            "weight_kg": positive,
            "price_usd_per_lb": positive,
            "total_usd": positive,
            "buyers": pa.Column(pl.String, nullable=True),
        },
    )


def roaster_origins_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of `roaster_origins`: every canonical value is one some other table uses."""
    sheets = rules.roaster_sheets
    in_vocabulary = {
        "country": sorted({*sheets.countries.values(), sheets.home_country}),
        "state": sorted(set(sheets.states.values())),
        "processing_method": sorted({*rules.processing_methods.values(), UNCLASSIFIED}),
        "species": sorted(set(sheets.species.values())),
    }
    optional_text = ("region", "producer", "farm", "process")
    return pa.DataFrameSchema(
        name="roaster_origins",
        strict=True,
        unique=["shop", "product_id", "origin"],
        columns={
            "coffee_id": pa.Column(pl.String),
            "shop": pa.Column(pl.String),
            "product_id": pa.Column(pl.String),
            "origin": pa.Column(pl.Int64, pa.Check.ge(1)),
            **{
                name: pa.Column(pl.String, pa.Check.isin(values), nullable=True)
                for name, values in in_vocabulary.items()
            },
            **{name: pa.Column(pl.String, nullable=True) for name in optional_text},
            "altitude_min_m": pa.Column(
                pl.Float64, pa.Check.in_range(*rules.altitude_m), nullable=True
            ),
            "altitude_max_m": pa.Column(
                pl.Float64, pa.Check.in_range(*rules.altitude_m), nullable=True
            ),
            "varieties": pa.Column(pl.List(pl.String), nullable=True),
            "sca_score": pa.Column(pl.Float64, pa.Check.in_range(0, 100), nullable=True),
        },
        checks=[
            pa.Check(
                lambda data: data.lazyframe.select(
                    pl.col("altitude_min_m").le(pl.col("altitude_max_m")).fill_null(True)
                ),
                error="an altitude range runs from its lowest point to its highest",
            )
        ],
    )


ROASTER_OFFERS = pa.DataFrameSchema(
    name="roaster_offers",
    strict=True,
    unique=["shop", "variant_id"],
    columns={
        "offer_id": pa.Column(pl.String, unique=True),  # "<shop>-<variant id>"
        "coffee_id": pa.Column(pl.String),
        "shop": pa.Column(pl.String),
        "product_id": pa.Column(pl.String),
        "variant_id": pa.Column(pl.String),
        "variant_title": pa.Column(pl.String, nullable=True),
        "price_mxn": pa.Column(pl.Float64, pa.Check.ge(0)),
        # Null when neither title states a size: never the platform's own weight.
        "bag_grams": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),
        "price_mxn_per_kg": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        # Listed as the shop lists it, but most likely a price copied from another size.
        "price_outlier": pa.Column(pl.Boolean, nullable=True),
        "observed_on": pa.Column(pl.Date),  # when the catalogue was read
        "snapshot": pa.Column(pl.String),
    },
)


# Every read of the catalogues, one row per offer per read: the latest read of a day
# stands for the day. The tables above are the catalogue as it is now.
ROASTER_OFFER_HISTORY = pa.DataFrameSchema(
    name="roaster_offer_history",
    strict=True,
    unique=["offer_id", "snapshot"],
    columns={
        "observation_id": pa.Column(pl.String, unique=True),  # "<offer id>@<snapshot>"
        **{name: column for name, column in ROASTER_OFFERS.columns.items() if name != "offer_id"},
        "offer_id": pa.Column(pl.String),  # the same offer, read again: not unique here
    },
)


def roaster_flavors_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """Contract of `roaster_flavors`: every note is one the lexicon holds, in its own
    category, and a coffee tastes of each note once."""
    groups = rules.tasting_notes.groups
    notes = sorted({note for group in groups for note in group.notes})
    subcategories = sorted({group.subcategory for group in groups if group.subcategory})
    return pa.DataFrameSchema(
        name="roaster_flavors",
        strict=True,
        unique=["coffee_id", "note_en"],
        columns={
            "coffee_id": pa.Column(pl.String),
            "shop": pa.Column(pl.String),
            "note": pa.Column(pl.String, pa.Check.isin(notes)),
            "note_en": pa.Column(pl.String),
            "category": pa.Column(pl.String, pa.Check.isin(rules.tasting_notes.categories)),
            "subcategory": pa.Column(pl.String, pa.Check.isin(subcategories), nullable=True),
            "source": pa.Column(pl.String, pa.Check.isin(["title", "description"])),
        },
    )


def roaster_origin_history_schema(rules: CleaningConfig) -> pa.DataFrameSchema:
    """`roaster_origins` for every read: a sheet can change between reads."""
    current = roaster_origins_schema(rules)
    return pa.DataFrameSchema(
        name="roaster_origin_history",
        strict=True,
        unique=["coffee_id", "origin", "snapshot"],
        columns={**current.columns, "snapshot": pa.Column(pl.String)},
        checks=current.checks,
    )


PRICE_INDICATORS = pa.DataFrameSchema(
    name="price_indicators",
    strict=True,
    unique=["period", "frequency", "indicator"],
    columns={
        # The day, or the first day of the month a monthly average is for.
        "period": pa.Column(pl.Date),
        "frequency": pa.Column(pl.String, pa.Check.isin(["daily", "monthly"])),
        "indicator": pa.Column(pl.String, pa.Check.isin(list(ICO_INDICATORS))),
        "usd_cents_per_lb": pa.Column(pl.Float64, pa.Check.gt(0)),
        "source": pa.Column(pl.String, pa.Check.isin(["ico", "world_bank"])),
        "read_at": pa.Column(pl.Datetime("us", "UTC")),  # the download the value came from
    },
)


# One price PROFECO recorded: a product in one presentation, on one shelf, on one day.
# A shop can have two prices for one product on one day (48 of 64,451 rows); both
# are kept, so the price is part of what makes a row.
CONSUMER_PRICES = pa.DataFrameSchema(
    name="consumer_prices",
    strict=True,
    unique=["date", "store", "latitude", "longitude", "brand", "presentation", "price_mxn"],
    columns={
        "date": pa.Column(pl.Date),
        # The fortnight PROFECO files it under: the 1st or the 16th of its month.
        "fortnight": pa.Column(
            pl.Date,
            pa.Check(
                lambda data: data.lazyframe.select(pl.col(data.key).dt.day().is_in([1, 16])),
                error="a fortnight starts on the 1st or the 16th",
            ),
        ),
        "product": pa.Column(pl.String, pa.Check.isin(["ground", "instant"])),
        "brand": pa.Column(pl.String),
        "presentation": pa.Column(pl.String),  # as PROFECO writes it
        "grams": pa.Column(pl.Float64, pa.Check.gt(0)),
        "sweetened": pa.Column(pl.Boolean),  # a blend with sugar or caramel
        "decaf": pa.Column(pl.Boolean),
        "price_mxn": pa.Column(pl.Float64, pa.Check.gt(0)),
        "price_mxn_per_kg": pa.Column(pl.Float64, pa.Check.gt(0)),
        "chain": pa.Column(pl.String),
        "store_type": pa.Column(pl.String),  # supermarket, convenience store, market...
        "store": pa.Column(pl.String),
        "state": pa.Column(pl.String),
        "municipality": pa.Column(pl.String),  # as the store declares it
        "latitude": pa.Column(pl.Float64),
        "longitude": pa.Column(pl.Float64),
        # The borough its store declares, by INEGI's key and name: the city's rows only.
        # Not where the coordinates fall - those put 7 of 120 stores in another borough.
        "borough_id": pa.Column(pl.String, nullable=True),
        "borough": pa.Column(pl.String, nullable=True),
    },
)


# Pesos per US dollar on each business day a rate was set.
EXCHANGE_RATES = pa.DataFrameSchema(
    name="exchange_rates",
    strict=True,
    unique=["date"],
    columns={
        "date": pa.Column(pl.Date),
        "mxn_per_usd": pa.Column(pl.Float64, pa.Check.gt(0)),
    },
)


# The national consumer price index, a month a row: base, the second half of July 2018 = 100.
CONSUMER_PRICE_INDEX = pa.DataFrameSchema(
    name="consumer_price_index",
    strict=True,
    unique=["month"],
    columns={
        "month": pa.Column(pl.Date),  # its first day
        "index": pa.Column(pl.Float64, pa.Check.gt(0)),
    },
)


METRO, METROBUS = "metro", "metrobus"
TRANSIT_SYSTEMS = [METRO, METROBUS]

# A Metro or Metrobús station: one per system, line and name, as the city's feed places
# it; a transfer is a station on each of its lines. Outside every borough (the State of
# Mexico's end of a line) it has no borough, and is kept.
TRANSIT_STATIONS = pa.DataFrameSchema(
    name="transit_stations",
    strict=True,
    unique=["station_id"],
    columns={
        "station_id": pa.Column(pl.String),  # "<system>-<line>-<name's words>"
        "system": pa.Column(pl.String, pa.Check.isin(TRANSIT_SYSTEMS)),
        "line": pa.Column(pl.String),
        "station": pa.Column(pl.String),
        "latitude": pa.Column(pl.Float64, pa.Check.in_range(-90, 90)),
        "longitude": pa.Column(pl.Float64, pa.Check.in_range(-180, 180)),
        "borough_id": pa.Column(pl.String, nullable=True),
        "borough": pa.Column(pl.String, nullable=True),
        "zone_id": pa.Column(pl.String, nullable=True),  # its urban AGEB
    },
)

# Entries a day: per Metro station, per Metrobús line (its counts name no station).
TRANSIT_RIDERSHIP = pa.DataFrameSchema(
    name="transit_ridership",
    strict=True,
    unique=["date", "system", "line", "station_id"],
    columns={
        "date": pa.Column(pl.Date),
        "system": pa.Column(pl.String, pa.Check.isin(TRANSIT_SYSTEMS)),
        "line": pa.Column(pl.String),
        "station_id": pa.Column(pl.String, nullable=True),  # null: a Metrobús line's day
        "station": pa.Column(pl.String, nullable=True),
        "entries": pa.Column(pl.Int64, pa.Check.ge(0)),  # zero: the station was closed
    },
)


# An urban AGEB: its polygon and what the 2020 Census counted in it. A figure INEGI
# withheld is null; population never is.
CENSUS_ZONES = pa.DataFrameSchema(
    name="census_zones",
    strict=True,
    unique=["zone_id"],
    columns={
        "zone_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{9}[0-9A-Z]{4}$")),
        "borough_id": pa.Column(pl.String, pa.Check.str_matches(r"^\d{5}$")),
        "borough": pa.Column(pl.String),
        "ageb": pa.Column(pl.String, pa.Check.str_matches(r"^[0-9A-Z]{3}-[0-9A-Z]$")),
        "label": pa.Column(pl.String),  # "Miguel Hidalgo · AGEB 048-2"
        "area_km2": pa.Column(pl.Float64, pa.Check.gt(0)),
        "boundary": pa.Column(pl.Binary),
        "population": pa.Column(pl.Int64, pa.Check.ge(0)),
        "dwellings": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "schooling_years": pa.Column(pl.Float64, pa.Check.in_range(0, 25), nullable=True),
        "economically_active": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "people_65_plus": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "dwellings_with_internet": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "dwellings_with_car": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
        "dwellings_with_computer": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
    },
)


# FAOSTAT's producer prices of coffee: a row per country (PSD's name) and year. Every
# price is per tonne; `cherry` marks the countries whose tonne is of cherry, checked.
PRODUCER_PRICES = pa.DataFrameSchema(
    name="producer_prices",
    strict=True,
    unique=["country", "year"],
    columns={
        "country": pa.Column(pl.String),
        "year": pa.Column(pl.Int64, pa.Check.in_range(1960, 2100)),
        "usd_per_t": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),
        "lcu_per_t": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),
        "price_index": pa.Column(pl.Float64, pa.Check.gt(0), nullable=True),  # 2014-2016 = 100
        "flag": pa.Column(pl.String, pa.Check.isin(["A", "B", "E", "I", "X"]), nullable=True),
        "cherry": pa.Column(pl.Boolean),
    },
)

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
