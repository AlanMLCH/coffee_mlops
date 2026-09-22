"""The roasters' shops, clean: their coffees, the origins each names, and its offers.

Three tables, because a shop describes three different things:

- `roaster_coffees`: one row per product a shop sells as coffee - the catalog's item.
- `roaster_origins`: one row per origin a product's sheet describes. Most name one; a
  blend lists each component in turn (Buna's Guarumbo: two arabicas and a robusta).
- `roaster_offers`: one row per product in one size, with its price per kilogram and
  the day it was observed: when the shops' catalogues were read.

`coffee_id` ("<shop>-<product id>") joins the three, and `offer_id` names an offer; a
platform's own ids are only unique within a shop.

The sheets are read by the core (`mlops_core.data.sheets`). What the labels mean, and
how a place, a variety or a process is written in Spanish, is `cleaning.roaster_sheets`
in the domain config, and every canonical value speaks the vocabulary of a table that
already exists: countries as PSD names them, states as SIAP does, processing methods and
varieties as the CQI does. That is what lets a 2026 bag from a Mexico City shop sit next
to a 2018 lot graded by the CQI.

Nothing is guessed. A size that is in neither title leaves the price per kilogram empty,
and the platform's own weight is never used: on Almanegra and Café con Jiribilla it
contradicts the size written in the variant's title. What a rule does not recognise is
kept as written and shows up in `analysis.roaster_coverage`.
"""

import logging
import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import polars as pl

from domains.coffee.config import OTHER, UNCLASSIFIED, CleaningConfig, RoasterSheetRules
from mlops_core.data.sheets import headed_paragraphs, html_text, labelled_lines, records

logger = logging.getLogger(__name__)

COFFEES = pl.Schema(
    {
        "coffee_id": pl.String,
        "shop": pl.String,
        "product_id": pl.String,
        "title": pl.String,
        "url": pl.String,
        "description": pl.String,  # the shop's own text, markup stripped: for search and RAG
        "origins": pl.Int64,  # how many origins its sheet describes; 0 = no sheet
    }
)
# Instances throughout: a List among bare classes would not type-check.
ORIGINS = pl.Schema(
    {
        "coffee_id": pl.String(),
        "shop": pl.String(),
        "product_id": pl.String(),
        "origin": pl.Int64(),  # 1, 2, ... in the order the sheet lists them
        "country": pl.String(),
        "state": pl.String(),
        "region": pl.String(),
        "producer": pl.String(),
        "farm": pl.String(),
        "altitude_min_m": pl.Float64(),
        "altitude_max_m": pl.Float64(),
        "varieties": pl.List(pl.String),
        "process": pl.String(),  # as the shop wrote it
        "processing_method": pl.String(),
        "species": pl.String(),
        "sca_score": pl.Float64(),
    }
)
OFFERS = pl.Schema(
    {
        "offer_id": pl.String,
        "coffee_id": pl.String,
        "shop": pl.String,
        "product_id": pl.String,
        "variant_id": pl.String,
        "variant_title": pl.String,
        "price_mxn": pl.Float64,
        "bag_grams": pl.Float64,  # everything in the offer: 12 bags of 340 g are 4,080 g
        "price_mxn_per_kg": pl.Float64,  # none for a bundle: its price pays for more
        # When the catalogue was read: the first ingestion of this exact content, so an
        # unchanged catalogue read again keeps its date.
        "observed_on": pl.Date,
        "snapshot": pl.String,  # that read, as the period the model's studies compare
    }
)

_TRIM = " .,;:-"
_PARENS = re.compile(r"\([^)]*\)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A share separates names like a comma does: "85% Typica 15% Caturra".
_SHARE = re.compile(r"\d+\s*%")
_LIST = re.compile(r",|&|\+|/|\by\b")  # "y" is Spanish for "and"
# "340 gr", "1 Kg.", "2kg", "250 GRAMOS", and Almanegra's "5/16 kg" (312.5 g).
_SIZE = re.compile(r"(\d+(?:\.\d+)?)(?:/(\d+))?\s*(kg|kilos?|gramos|grs?|g)\b", re.IGNORECASE)
_GRAMS_PER_UNIT = {"kg": 1000.0, "kilo": 1000.0, "kilos": 1000.0}  # anything else is grams
# How many bags an offer holds: "3 x 340g", "Caja de 12 Bolsas".
_PACK = re.compile(r"(\d+)\s*(?:x\b|bolsas\b)", re.IGNORECASE)
SCA_RANGE = (0.0, 100.0)  # the cupping scale; anything outside it is not a score


def fold(text: str) -> str:
    """Lower-case and accent-free: "Etiopía" and "ETIOPIA" are one key."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def place(text: str, rules: RoasterSheetRules) -> tuple[str | None, str | None]:
    """(country, state) a place names, reading its comma-separated parts whole.

    Whole parts, not words: "Pluma Hidalgo" is a town in Oaxaca, not the state of
    Hidalgo, and "Nuevo México, Chiapas" is in Chiapas.
    """
    country = state = None
    for part in text.split(","):
        key = fold(part).strip(_TRIM)
        country = country or rules.countries.get(key)
        state = state or rules.states.get(key)
    return country, state


def altitude_range(text: str, bounds: tuple[float, float]) -> tuple[float | None, float | None]:
    """'1,200 - 1,700 msnm' -> (1200, 1700).

    Commas are thousands separators here. A number outside the plausible range is not
    an altitude: in the typo '1,1150 msnm' the 1 is dropped and 1150 kept.
    """
    numbers = (float(n) for n in _NUMBER.findall(_THOUSANDS.sub("", text)))
    plausible = [n for n in numbers if bounds[0] <= n <= bounds[1]]
    return (min(plausible), max(plausible)) if plausible else (None, None)


def variety_list(text: str, aliases: Mapping[str, str]) -> list[str]:
    """'Typica, Bourbon y Marsellesa' -> ['bourbon', 'marsellesa', 'typica'], spelled
    as the CQI spells them where it has them."""
    parts = _LIST.split(_SHARE.sub(",", _PARENS.sub(" ", fold(text))))
    names = (" ".join(part.split()).strip(_TRIM) for part in parts)
    return sorted({aliases.get(name, name) for name in names if name})


def processing_method(label: str, rules: RoasterSheetRules) -> str:
    """The CQI's processing method a process label names.

    Experimental fermentations are "other", as the CQI rules have them. So is a label
    naming two methods: a lot sold as "Lavado, Natural" was not processed one way.
    """
    text = _PARENS.sub(" ", fold(label))
    if re.search(rules.experimental, text):
        return OTHER
    found = []
    for method, pattern in rules.processes.items():
        if re.search(pattern, text):
            found.append(method)
            text = re.sub(pattern, " ", text)  # "semi-lavado" must not also be "lavado"
    if len(found) == 1:
        return found[0]
    return OTHER if found else UNCLASSIFIED


def bag_grams(variant_title: str | None, title: str) -> float | None:
    """Grams in an offer, from its titles: the variant's size, else the product's,
    times the bags in a pack. None when neither says."""
    size = _size(variant_title or "") or _size(title)
    if size is None:
        return None
    return size * (_pack(title) or _pack(variant_title or "") or 1)


def clean_roasters(
    offers: pl.DataFrame | None, rules: CleaningConfig, read_at: datetime | None = None
) -> dict[str, pl.DataFrame]:
    """The raw offers, read at `read_at` -> `roaster_coffees`, `roaster_origins` and
    `roaster_offers`.

    `offers` is None when the shops were never read: the tables are then empty, with
    their columns, and everything else still builds.
    """
    if offers is None:
        logger.info("roaster_catalogs was never ingested: its clean tables are empty")
        offers = pl.DataFrame()
    elif read_at is None:
        raise ValueError("Offers need the time their catalogue was read")
    sheets = rules.roaster_sheets
    coffees, origins = [], []
    unmapped: set[str] = set()
    products = (
        offers.unique(["shop", "product_id"], keep="first", maintain_order=True)
        if not offers.is_empty()
        else offers
    )
    for product in products.iter_rows(named=True):
        described = _sheet(product, sheets)
        key = {
            "coffee_id": f"{product['shop']}-{product['product_id']}",
            "shop": product["shop"],
            "product_id": product["product_id"],
        }
        for number, record in enumerate(described, start=1):
            origin = _origin(record, rules)
            if "country" in record and origin["country"] is None:
                unmapped.add(record["country"])
            origins.append({**key, "origin": number, **origin})
        coffees.append(
            {
                **key,
                "title": product["title"],
                "url": product["url"],
                "description": html_text(product["body_html"]) or None,
                "origins": len(described),
            }
        )
    if unmapped:
        logger.warning(
            "Countries no rule maps, left empty: %s. Add them to cleaning.roaster_sheets.countries",
            sorted(unmapped),
        )
    return {
        "roaster_coffees": pl.DataFrame(coffees, schema=COFFEES).sort("shop", "product_id"),
        "roaster_origins": pl.DataFrame(origins, schema=ORIGINS).sort(
            "shop", "product_id", "origin"
        ),
        "roaster_offers": _priced(offers, sheets, read_at),
    }


def _sheet(product: Mapping[str, Any], sheets: RoasterSheetRules) -> list[dict[str, str]]:
    """The origins a product's sheet describes: from its page's headed paragraphs where
    the shop keeps them there, else from the "Label: value" lines of its description."""
    pairs = headed_paragraphs(product["page_html"], sheets.labels) or labelled_lines(
        html_text(product["body_html"]), sheets.labels
    )
    return records(pairs)


def _origin(record: Mapping[str, str], rules: CleaningConfig) -> dict[str, Any]:
    """One origin's canonical values; what a rule does not recognise stays empty."""
    sheets = rules.roaster_sheets
    country = state = None
    for field in ("country", "state", "origin", "region"):  # the most specific first
        if field in record:
            named_country, named_state = place(record[field], sheets)
            country, state = country or named_country, state or named_state
    if state is not None and country is None:
        country = sheets.home_country
    low, high = altitude_range(record.get("altitude", ""), rules.altitude_m)
    process = _trimmed(record.get("process"))
    varieties = variety_list(record["varieties"], sheets.varieties) if "varieties" in record else []
    return {
        "country": country,
        "state": state,
        "region": _trimmed(record.get("region") or record.get("origin")),
        "producer": _trimmed(record.get("producer")),
        "farm": _trimmed(record.get("farm")),
        "altitude_min_m": low,
        "altitude_max_m": high,
        "varieties": varieties or None,
        "process": process,
        "processing_method": processing_method(process, sheets) if process else None,
        "species": sheets.species.get(fold(record.get("species", "")).strip(_TRIM)),
        "sca_score": _score(record.get("sca_score")),
    }


def _priced(
    offers: pl.DataFrame, sheets: RoasterSheetRules, read_at: datetime | None
) -> pl.DataFrame:
    """Each offer with the grams its titles state, the price per kilogram, and whether
    that price is so far from the product's other offers that it was entered wrong.

    The price stays as listed - it is what the shop charges - and is flagged, not fixed.
    """
    rows = []
    contradicted = 0
    for offer in offers.iter_rows(named=True):
        grams = bag_grams(offer["variant_title"], offer["title"])
        by_weight = grams and not re.search(sheets.bundles, fold(offer["title"]))
        platform = offer["platform_grams"]
        if grams and platform and abs(platform - grams) > 0.05 * grams:
            contradicted += 1
        rows.append(
            {
                "offer_id": f"{offer['shop']}-{offer['variant_id']}",
                "coffee_id": f"{offer['shop']}-{offer['product_id']}",
                "shop": offer["shop"],
                "product_id": offer["product_id"],
                "variant_id": offer["variant_id"],
                "variant_title": offer["variant_title"],
                "price_mxn": offer["price"],
                "bag_grams": grams,
                "price_mxn_per_kg": round(offer["price"] / grams * 1000, 2) if by_weight else None,
                "observed_on": read_at.date() if read_at else None,
                "snapshot": read_at.date().isoformat() if read_at else None,
            }
        )
    if contradicted:
        # Packaging accounts for some (12 bags of 340 g ship as 4.4 kg); a wrong entry
        # for the rest. Either way the title is what the buyer is sold.
        logger.info("The platform's weight differs from the titles in %d offers", contradicted)
    ratio = pl.col("price_mxn_per_kg") / pl.col("price_mxn_per_kg").median().over(
        "shop", "product_id"
    )
    limit = sheets.price_outlier_ratio
    priced = (
        pl.DataFrame(rows, schema=OFFERS)
        .with_columns(price_outlier=(ratio > limit) | (ratio < 1 / limit))
        .sort("shop", "product_id", "variant_id")
    )
    if outliers := priced["price_outlier"].sum():
        logger.warning("%d offers priced far off their product's other offers", outliers)
    return priced


def _size(text: str) -> float | None:
    match = _SIZE.search(text)
    if match is None:
        return None
    amount = float(match.group(1)) / float(match.group(2) or 1)
    return amount * _GRAMS_PER_UNIT.get(match.group(3).lower(), 1.0)


def _pack(text: str) -> int | None:
    match = _PACK.search(text)
    return int(match.group(1)) if match else None


def _score(text: str | None) -> float | None:
    number = _NUMBER.search(text or "")
    score = float(number.group()) if number else None
    return score if score is not None and SCA_RANGE[0] <= score <= SCA_RANGE[1] else None


def _trimmed(text: str | None) -> str | None:
    return (text or "").strip(_TRIM) or None
