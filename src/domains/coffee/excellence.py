"""Cup of Excellence Mexico: the country's best lots each year, judged blind by a national
and an international jury, and what an online auction paid for them.

The page of each year gives its lots and its sales in separate tables (see
`domains.coffee.sources.coe`); here they become one row per lot. A sale finds the one lot
of its year with its rank, score and farm, else its rank and score, else its rank (before
2021 the auction gave no score), else its farm and score (the national winners have no
rank). A score matches to within 0.05 (the 2026 auction rounds 87.56 to 87.6) and a farm
when one name holds the other ("La Bendición", "Finca La Bendición"): 2024 and later ran
three competitions a year, each with its own fifth place, two of them at 87.78. A lot no
sale finds was not sold. Two sales of one lot, or a sale of no lot, stop the build: the
pairing would be guessing.

The place is read as the roasters' sheets are read (the coffee-growing states' names in
the region's parts); a region that names only a municipality ("San Juan Lachao") is
placed by SIAP's municipalities, when only one state has a municipality of that name.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from domains.coffee.config import RoasterSheetRules
from domains.coffee.roaster_sheets import fold, place, processing_method, variety_list

logger = logging.getLogger(__name__)

POUNDS_PER_KG = 2.20462
NATIONAL_WINNER = "nw"
SCORE_ROUNDING = 0.05 + 1e-9
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A comma before one or two digits at the end is a decimal point ("88,11", 2025).
_DECIMAL_COMMA = re.compile(r"^(\d+),(\d{1,2})$")


def clean_cup_of_excellence(
    raw: pl.DataFrame, rules: RoasterSheetRules, production: pl.DataFrame
) -> pl.DataFrame:
    """A row per lot of every year's competition: its score, farm, place, varieties and
    process, and its sale - US dollars a pound, the total and who bought it - if sold."""
    rows = raw.to_dicts()
    lots = [row for row in rows if row["farmer"] or row["region"]]
    sales = [row for row in rows if row["bid"]]
    sold = _paired(lots, sales)
    municipalities = _municipality_states(production)
    records = []
    numbered: dict[str, int] = defaultdict(int)  # a lot's number in its year, in page order
    for lot in lots:
        sale = sold.get(id(lot), {})
        price, total = _number(sale.get("bid")), _number(sale.get("total"))
        # Before 2018 a lot's size is in boxes: its pounds are what it fetched over its price.
        paid = round(total / price / POUNDS_PER_KG, 1) if price and total else None
        weight = _weight(lot) or _weight(sale) or paid
        records.append(
            {
                "year": int(lot["year"]),
                "lot_id": f"{lot['year']}-{numbered[lot['year']]:03d}",
                "rank": _rank(lot),
                "national_winner": _rank(lot) in (None, NATIONAL_WINNER),
                "score": _number(lot["score"]),
                "farm": lot["farm"],
                "farmer": lot["farmer"],
                "region": lot["region"],
                "state": _state(lot["region"], rules, municipalities),
                "varieties": variety_list(lot["variety"], rules.varieties)
                if lot["variety"]
                else None,
                "processing_method": processing_method(lot["process"], rules)
                if lot["process"]
                else None,
                "weight_kg": weight,
                "price_usd_per_lb": price,
                "total_usd": total,
                "buyers": sale.get("buyers"),
            }
        )
        numbered[lot["year"]] += 1
    return pl.DataFrame(records, schema=SCHEMA)


SCHEMA = pl.Schema(
    {
        "year": pl.Int64(),
        "lot_id": pl.String(),
        "rank": pl.String(),
        "national_winner": pl.Boolean(),
        "score": pl.Float64(),
        "farm": pl.String(),
        "farmer": pl.String(),
        "region": pl.String(),
        "state": pl.String(),
        "varieties": pl.List(pl.String),
        "processing_method": pl.String(),
        "weight_kg": pl.Float64(),
        "price_usd_per_lb": pl.Float64(),
        "total_usd": pl.Float64(),
        "buyers": pl.String(),
    }
)


def _paired(lots: list[dict[str, Any]], sales: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Each sold lot (by identity) -> its sale."""
    by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lot in lots:
        by_year[lot["year"]].append(lot)
    sold: dict[int, dict[str, Any]] = {}
    unpaired: list[str] = []
    for sale in sales:
        of = _lot_of(sale, by_year[sale["year"]])
        if of is None:
            unpaired.append(f"{sale['year']} {sale['rank']} {sale['farm']}")
        elif id(of) in sold:
            raise ValueError(f"Two sales of one lot: {sale['year']} {of['farm']}")
        else:
            sold[id(of)] = sale
    if unpaired:
        raise ValueError(f"Sales of no lot on the page: {unpaired}")
    return sold


def _lot_of(sale: dict[str, Any], lots: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one lot of the year the sale is of, or None."""
    if sale["farmer"] or sale["region"]:
        return sale  # one row holds both
    rank, score, farm = _rank(sale), _number(sale["score"]), _key(sale["farm"])

    def scored(lot: dict[str, Any]) -> bool:
        theirs = _number(lot["score"])
        return score is not None and theirs is not None and abs(theirs - score) <= SCORE_ROUNDING

    def named(lot: dict[str, Any]) -> bool:
        theirs = _key(lot["farm"])
        return bool(farm and theirs) and (farm in theirs or theirs in farm)

    tries: list[Callable[[dict[str, Any]], bool]] = []
    if rank not in (None, NATIONAL_WINNER):
        tries += [
            lambda lot: _rank(lot) == rank and scored(lot) and named(lot),
            lambda lot: _rank(lot) == rank and scored(lot),
            lambda lot: _rank(lot) == rank,
        ]
    tries.append(lambda lot: scored(lot) and named(lot))
    for matches in tries:
        found = [lot for lot in lots if matches(lot)]
        if len(found) == 1:
            return found[0]
    return None


def _rank(row: dict[str, Any]) -> str | None:
    rank = row.get("rank")
    return rank.strip().lower() if rank else None


def _key(farm: str | None) -> str:
    """A farm's words, folded: "Rancho Viejo - Kohmar" is "Rancho viejo-Kohmar"."""
    return " ".join(re.findall(r"[a-z0-9]+", fold(farm or "")))


def _number(text: str | None) -> float | None:
    """'$50.21/lb', 'US$ 61.00', '1,388.91', '93' -> the number; '' or None -> None."""
    found = _NUMBER.search(text or "")
    if found is None:
        return None
    number = _DECIMAL_COMMA.sub(r"\1.\2", found.group())
    return float(number.replace(",", ""))


def _weight(row: dict[str, Any]) -> float | None:
    kg = _number(row.get("weight_kg"))
    if kg is not None:
        return kg
    pounds = _number(row.get("weight_lb"))
    return round(pounds / POUNDS_PER_KG, 1) if pounds is not None else None


def _municipality_states(production: pl.DataFrame) -> dict[str, str]:
    """A municipality's name (folded) -> its state, for the names only one state has."""
    states = production.group_by(
        pl.col("municipality").map_elements(fold, return_dtype=pl.String).alias("key")
    ).agg(pl.col("state").unique())
    return {row["key"]: row["state"][0] for row in states.iter_rows(named=True)
            if len(row["state"]) == 1}  # fmt: skip


def _state(
    region: str | None, rules: RoasterSheetRules, municipalities: dict[str, str]
) -> str | None:
    if not region:
        return None
    _, state = place(region, rules)
    if state is not None:
        return state
    found = {municipalities[key] for part in region.split(",")
             if (key := " ".join(fold(part).split())) in municipalities}  # fmt: skip
    return found.pop() if len(found) == 1 else None


def coe_by_year(
    lots: pl.DataFrame, indicators: pl.DataFrame, rates: pl.DataFrame, commodity: str
) -> pl.DataFrame:
    """Each year's competition: its lots, how many sold, their scores and auction prices,
    and the median price against the year's commodity price (`commodity`: the monthly
    indicator of the group Mexico's coffee is priced in) and in pesos a kilogram."""
    market = (
        indicators.filter((pl.col("frequency") == "monthly") & (pl.col("indicator") == commodity))
        .group_by(pl.col("period").dt.year().cast(pl.Int64).alias("year"))
        .agg((pl.col("usd_cents_per_lb").mean() / 100).alias("commodity_usd_per_lb"))
    )
    pesos = rates.group_by(pl.col("date").dt.year().cast(pl.Int64).alias("year")).agg(
        pl.col("mxn_per_usd").mean()
    )
    price = pl.col("price_usd_per_lb")
    return (
        lots.group_by("year")
        .agg(
            pl.len().alias("lots"),
            price.is_not_null().sum().alias("sold"),
            pl.col("score").median().alias("median_score"),
            price.median().alias("median_usd_per_lb"),
            price.max().alias("top_usd_per_lb"),
            pl.col("total_usd").sum().alias("auction_usd"),
        )
        .join(market, on="year", how="left")
        .join(pesos, on="year", how="left")
        .with_columns(
            (pl.col("median_usd_per_lb") / pl.col("commodity_usd_per_lb")).alias(
                "median_times_commodity"
            ),
            (pl.col("median_usd_per_lb") * POUNDS_PER_KG * pl.col("mxn_per_usd")).alias(
                "median_mxn_per_kg"
            ),
        )
        .sort("year")
    )


def coe_score_price(lots: pl.DataFrame, resamples: int = 1000, seed: int = 7) -> pl.DataFrame:
    """What a point of score is worth at auction: the slope of the log price on the score
    with each year its own level (prices rose year on year), as the percent a point adds,
    and the rank correlation within years; 95% intervals from resampling the lots."""
    sold = lots.drop_nulls(["score", "price_usd_per_lb"]).sort("lot_id")
    years = sold["year"].to_numpy()
    score = sold["score"].to_numpy()
    log_price = np.log(sold["price_usd_per_lb"].to_numpy())

    def slope(idx: np.ndarray) -> float:
        dummies = (years[idx][:, None] == np.unique(years[idx])[None, :]).astype(float)
        design = np.column_stack([score[idx], dummies])
        return float(np.linalg.lstsq(design, log_price[idx], rcond=None)[0][0])

    def within(idx: np.ndarray) -> float:
        frame = pl.DataFrame({"year": years[idx], "score": score[idx], "price": log_price[idx]})
        ranked = frame.with_columns(pl.col("score", "price").rank().over("year")).select(
            pl.corr("score", "price")
        )
        return float(ranked.item())

    everything = np.arange(len(sold))
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(sold), len(sold)) for _ in range(resamples)]
    slopes = [slope(d) for d in draws]
    rhos = [within(d) for d in draws]
    return pl.DataFrame(
        {
            "lots": [len(sold)],
            "years": [int(np.unique(years).size)],
            "percent_per_point": [_percent(slope(everything))],
            "percent_low": [_percent(np.percentile(slopes, 2.5))],
            "percent_high": [_percent(np.percentile(slopes, 97.5))],
            "rank_correlation": [within(everything)],
            "rank_correlation_low": [float(np.percentile(rhos, 2.5))],
            "rank_correlation_high": [float(np.percentile(rhos, 97.5))],
        }
    )


def _percent(slope: float) -> float:
    """A slope of the log price as the percent one more point adds."""
    return float(100 * (np.exp(slope) - 1))
