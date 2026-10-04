"""The coffee shop's contracts: the point-of-sale export as it arrives (raw), and the
canonical tables a shop's data becomes (clean).

The raw contracts are a point-of-sale export's: text, as any system would write it, with
the shape checked before a value is read. The clean ones are the domain's promise to
every reader - the models, the studies, the agent - whatever system the data came from.
"""

import pandera.polars as pa
import polars as pl

_DATE = r"^\d{4}-\d{2}-\d{2}$"
_MOMENT = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$"
_NUMBER = r"^-?\d+(\.\d+)?$"

# A real coffee machine's sales (CC0), the pattern the shop borrows: one sale a row.
VENDING_SALES = pa.DataFrameSchema(
    name="vending_sales",
    columns={
        "date": pa.Column(pl.String, pa.Check.str_matches(_DATE)),
        "datetime": pa.Column(pl.String, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2} \d{2}:")),
        "money": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
        "coffee_name": pa.Column(pl.String),
    },
)


def _export(name: str, columns: dict[str, pa.Column]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(name=name, strict=True, columns=columns)


RAW_SCHEMAS = {
    "vending_sales": VENDING_SALES,
    "pos_sales": _export(
        "pos_sales",
        {
            "ticket": pa.Column(pl.String),
            "line": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
            "sold_at": pa.Column(pl.String, pa.Check.str_matches(_MOMENT)),
            "product": pa.Column(pl.String),
            "quantity": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
            "unit_price": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
            "channel": pa.Column(pl.String, pa.Check.isin(["dine_in", "takeaway"])),
            "payment": pa.Column(pl.String, pa.Check.isin(["card", "cash"])),
        },
    ),
    "pos_orders": _export(
        "pos_orders",
        {
            "ticket": pa.Column(pl.String),
            "ordered_at": pa.Column(pl.String, pa.Check.str_matches(_MOMENT)),
            "ready_at": pa.Column(pl.String, pa.Check.str_matches(_MOMENT)),
            "items": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
            "baristas": pa.Column(pl.String, pa.Check.str_matches(r"^\d+$")),
        },
    ),
    "pos_menu": _export(
        "pos_menu",
        {
            "product": pa.Column(pl.String),
            "category": pa.Column(pl.String),
            "price": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
            "valid_from": pa.Column(pl.String, pa.Check.str_matches(_DATE)),
            "valid_to": pa.Column(pl.String, pa.Check.str_matches(_DATE), nullable=True),
        },
    ),
    "pos_purchases": _export(
        "pos_purchases",
        {
            "date": pa.Column(pl.String, pa.Check.str_matches(_DATE)),
            "ingredient": pa.Column(pl.String),
            "unit": pa.Column(pl.String),
            "quantity": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
            "cost": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
        },
    ),
    "pos_shifts": _export(
        "pos_shifts",
        {
            "date": pa.Column(pl.String, pa.Check.str_matches(_DATE)),
            "staff": pa.Column(pl.String),
            "role": pa.Column(pl.String),
            "starts_at": pa.Column(pl.String, pa.Check.str_matches(_MOMENT)),
            "ends_at": pa.Column(pl.String, pa.Check.str_matches(_MOMENT)),
            "hourly_wage": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
        },
    ),
    "pos_recipes": _export(
        "pos_recipes",
        {
            "product": pa.Column(pl.String),
            "ingredient": pa.Column(pl.String),
            "quantity": pa.Column(pl.String, pa.Check.str_matches(_NUMBER)),
            "unit": pa.Column(pl.String),
        },
    ),
}


def _clean(name: str, unique: list[str], columns: dict[str, pa.Column]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(name=name, strict=True, unique=unique, columns=columns)


_MONEY = pa.Column(pl.Float64, pa.Check.ge(0))
_MOMENT_COLUMN = pa.Column(pl.Datetime("us"))

CLEAN_SCHEMAS = {
    # One line of a ticket: a product sold, at the price it was sold at.
    "sales": _clean(
        "sales",
        ["ticket_id", "line"],
        {
            "ticket_id": pa.Column(pl.String),
            "line": pa.Column(pl.Int64, pa.Check.ge(1)),
            "sold_at": _MOMENT_COLUMN,
            "date": pa.Column(pl.Date),
            "hour": pa.Column(pl.Int64, pa.Check.in_range(0, 23)),
            "weekday": pa.Column(pl.Int64, pa.Check.in_range(1, 7)),
            "product": pa.Column(pl.String),
            "category": pa.Column(pl.String),
            "quantity": pa.Column(pl.Int64, pa.Check.ge(1)),
            "unit_price_mxn": _MONEY,
            "line_total_mxn": _MONEY,
            "channel": pa.Column(pl.String),
            "payment": pa.Column(pl.String),
        },
    ),
    # One ticket: when it was ordered, when it was ready, and the hands there were.
    "orders": _clean(
        "orders",
        ["ticket_id"],
        {
            "ticket_id": pa.Column(pl.String),
            "ordered_at": _MOMENT_COLUMN,
            "ready_at": _MOMENT_COLUMN,
            "date": pa.Column(pl.Date),
            "hour": pa.Column(pl.Int64, pa.Check.in_range(0, 23)),
            "minutes": pa.Column(pl.Float64, pa.Check.ge(0)),  # from order to ready
            "items": pa.Column(pl.Int64, pa.Check.ge(1)),
            "baristas": pa.Column(pl.Int64, pa.Check.ge(1)),
        },
    ),
    # A product's price over a period: the menu's history.
    "menu_prices": _clean(
        "menu_prices",
        ["product", "valid_from"],
        {
            "product": pa.Column(pl.String),
            "category": pa.Column(pl.String),
            "price_mxn": _MONEY,
            "valid_from": pa.Column(pl.Date),
            "valid_to": pa.Column(pl.Date, nullable=True),
        },
    ),
    # What went into one of a product: the recipe.
    "recipes": _clean(
        "recipes",
        ["product", "ingredient"],
        {
            "product": pa.Column(pl.String),
            "ingredient": pa.Column(pl.String),
            "quantity": pa.Column(pl.Float64, pa.Check.gt(0)),
            "unit": pa.Column(pl.String),
        },
    ),
    # One purchase of an ingredient.
    "purchases": _clean(
        "purchases",
        ["date", "ingredient"],
        {
            "date": pa.Column(pl.Date),
            "ingredient": pa.Column(pl.String),
            "unit": pa.Column(pl.String),
            "quantity": pa.Column(pl.Float64, pa.Check.gt(0)),
            "cost_mxn": _MONEY,
            "unit_cost_mxn": _MONEY,
        },
    ),
    # One shift worked.
    "shifts": _clean(
        "shifts",
        ["date", "staff_id"],
        {
            "date": pa.Column(pl.Date),
            "staff_id": pa.Column(pl.String),
            "role": pa.Column(pl.String),
            "starts_at": _MOMENT_COLUMN,
            "ends_at": _MOMENT_COLUMN,
            "hours": pa.Column(pl.Float64, pa.Check.gt(0)),
            "hourly_wage_mxn": _MONEY,
            "cost_mxn": _MONEY,
        },
    ),
    # One hour the shop was open: what it sold, at what prices, with how many hands.
    "shop_hours": _clean(
        "shop_hours",
        ["hour_id"],
        {
            "hour_id": pa.Column(pl.String),
            "date": pa.Column(pl.Date),
            "month": pa.Column(pl.String),
            "hour": pa.Column(pl.Int64, pa.Check.in_range(0, 23)),
            "weekday": pa.Column(pl.Int64, pa.Check.in_range(1, 7)),
            "tickets": pa.Column(pl.Float64, pa.Check.ge(0)),
            "items": pa.Column(pl.Float64, pa.Check.ge(0)),
            "revenue_mxn": _MONEY,
            "price_level": pa.Column(pl.Float64, pa.Check.gt(0)),
            "baristas": pa.Column(pl.Int64, pa.Check.ge(0)),
            "mean_minutes": pa.Column(pl.Float64, nullable=True),  # order to ready
        },
    ),
}
