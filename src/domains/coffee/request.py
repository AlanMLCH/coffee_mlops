"""What the prediction API accepts for coffee: a lot before it is cupped, a bag on a
shelf, a month whose green coffee price is not out yet - and since v1.1.1 a zone of the
city, a price some months ahead, a jar in a supermarket, a place by its name, a lot at
auction and a household."""

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _lower(value: object) -> object:
    """A closed vocabulary is lower case ("gesha", "washed", "almanegra"); "Gesha" is the
    same word, and the model would otherwise take it for a category it never saw."""
    return value.strip().lower() if isinstance(value, str) else value


# The vocabularies the models learned, said where a caller - or the agent's model, which
# reads these descriptions - chooses the words.
COUNTRY = "The origin country as USDA PSD names it, e.g. Ethiopia, Colombia, United States"
METHODS = "washed, natural, honey, semi_washed or other"


class Lot(BaseModel):
    """What a caller knows about a green coffee lot before it is cupped."""

    country: str = Field(description=COUNTRY)
    variety: str | None = Field(
        None,
        description="Lower case, as the CQI spells it, e.g. caturra, bourbon, typica, gesha, "
        "sl28, ethiopian heirlooms",
    )
    processing_method: str | None = Field(None, description=METHODS)
    color: str | None = Field(
        None, description="The green beans: green, blue-green, yellow-green, yellowish, brownish"
    )
    altitude_m: float | None = Field(None, ge=0, le=9000, description="Metres above sea level")
    moisture_pct: float | None = Field(None, ge=0, le=100, description="Of the green beans, %")
    category_one_defects: int = Field(0, ge=0, description="Primary (visual) defects")
    category_two_defects: int = Field(0, ge=0, description="Secondary defects")
    quakers: int | None = Field(None, ge=0, description="Unripe beans that fail to roast")
    graded_on: date | None = Field(None, description="Defaults to today (UTC).")

    _lowered = field_validator("variety", "processing_method", "color", mode="before")(_lower)

    def to_item(self) -> dict[str, Any]:
        """The lot as a row of the item table: `graded_on` is its time column."""
        return self.model_dump(exclude={"graded_on"}) | {
            "grading_date": self.graded_on or datetime.now(UTC).date()
        }


class Offer(BaseModel):
    """A bag of roasted coffee as a shop would list it, to price it per kilogram."""

    shop: str = Field(
        description="The roaster, as the catalogues name it: almanegra, buna, cucurucho or "
        "jiribilla (Café con Jiribilla)"
    )
    bag_grams: float = Field(gt=0, le=20_000, description="The bag's size, grams")
    country: str | None = Field(None, description=COUNTRY)
    state: str | None = Field(
        None, description="For a Mexican coffee, the state as SIAP names it, e.g. Oaxaca, Chiapas"
    )
    processing_method: str | None = Field(None, description=METHODS)
    variety: str | None = Field(
        None, description="Lower case, as the CQI spells it, e.g. gesha, typica, bourbon"
    )
    producer: str | None = Field(None, description="The farm or producer, as the sheet names it")
    altitude_m: float | None = Field(None, ge=0, le=9000, description="Metres above sea level")
    observed_on: date | None = Field(None, description="Defaults to today (UTC).")

    _lowered = field_validator("shop", "processing_method", "variety", mode="before")(_lower)

    def to_item(self) -> dict[str, Any]:
        """The bag as a row of the offers table: `observed_on` is its time column."""
        return self.model_dump(exclude={"observed_on"}) | {
            "observed_on": self.observed_on or datetime.now(UTC).date()
        }


class PriceMonth(BaseModel):
    """A month to forecast an international green coffee price for: its change from the
    month before, which has to be published already."""

    indicator: Literal["other_milds", "robustas"] = Field(
        description="other_milds (other mild Arabicas) or robustas, as the World Bank averages them"
    )
    month: date = Field(description="Any day of the month to forecast")

    def to_item(self) -> dict[str, Any]:
        """The month as a row of the price table, its own price not known yet."""
        return {
            "period": self.month.replace(day=1),
            "frequency": "monthly",
            "indicator": self.indicator,
            "usd_cents_per_lb": None,
        }


class Zone(BaseModel):
    """A city block group: how many coffee shops a zone like it has."""

    zone_id: str = Field(
        pattern=r"^09\d{7}[0-9A-Z]{4}$",
        # What the key is and its form, nothing more. Shown an example key, or told it
        # comes "from the 2020 Census", the 4B wrote "null": the question names no census
        # (2026-10-03).
        description="The urban AGEB key, 13 characters",
    )

    def to_item(self) -> dict[str, Any]:
        """The zone as a row of `census_zones`: it is looked up by its key."""
        return {"zone_id": self.zone_id}


class PriceOutlook(BaseModel):
    """An international green coffee price some months after the last one published."""

    indicator: Literal["other_milds", "robustas"] = Field(
        description="other_milds (other mild Arabicas) or robustas, as the World Bank averages them"
    )
    months_ahead: Literal[3, 6, 12] = Field(
        description="How far after the last published month: 3, 6 or 12"
    )

    def to_item(self) -> dict[str, Any]:
        """A month of the price table with no date and no price yet: it is the indicator's
        last published month plus `months_ahead`."""
        return {
            "period": None,
            "frequency": "monthly",
            "indicator": self.indicator,
            "usd_cents_per_lb": None,
            "horizon_months": self.months_ahead,
        }


class ShelfItem(BaseModel):
    """A jar or bag of coffee on a Mexican shelf, to price it fairly."""

    brand: str = Field(
        description="As PROFECO names it: Nescafé Clásico, Nescafé Dolca, Nescafé Decaf, Legal, "
        "Internacional, Internacional Americano, Los Portales, Los Portales de Córdoba, Great "
        "Value, Precíssimo or Chedraui"
    )
    grams: float = Field(gt=0, le=2_000, description="The jar's or bag's size, grams")
    product: Literal["instant", "ground"] = Field(
        description="instant or ground (roast and ground)"
    )
    sweetened: bool = Field(False, description="Mixed with sugar or caramel")
    decaf: bool = Field(False, description="Decaffeinated")
    chain: str | None = Field(
        None,
        description="The store's chain, e.g. Wal-mart, Bodega Aurrera, Hipermercado Soriana, "
        "Chedraui, La Comer, Oxxo, Superissste",
    )
    store_type: str | None = Field(
        None,
        description="Supermercado / Tienda de Autoservicio, Tienda de Conveniencia, Mercados, "
        "Farmacias or Central de Abasto",
    )
    state: str | None = Field(None, description="The Mexican state, e.g. Ciudad de México, Jalisco")
    observed_on: date | None = Field(None, description="Defaults to today (UTC).")

    def to_item(self) -> dict[str, Any]:
        """A reading of the survey without a price: what the shelf would ask is predicted."""
        return self.model_dump(exclude={"observed_on"}) | {
            "store": "request",
            "presentation": None,
            "date": self.observed_on or datetime.now(UTC).date(),
            "price_mxn": None,
        }


class Place(BaseModel):
    """A place DENUE lists with coffee shops, ice cream and juice bars, by its name."""

    name: str = Field(min_length=1, description="The place's name, as its sign or DENUE gives it")
    employees_band: str | None = Field(
        None,
        description="DENUE's staff band: 0 a 5 personas, 6 a 10 personas, 11 a 30 personas, "
        "31 a 50 personas, 51 a 100 personas",
    )
    zone_id: str | None = Field(
        None,
        pattern=r"^09\d{7}[0-9A-Z]{4}$",
        description="The urban AGEB key of where it stands, 13 characters",
    )

    def to_item(self) -> dict[str, Any]:
        """The place as a row of `coffee_shops` whose name no rule classified."""
        return self.model_dump() | {
            "shop_id": "request",
            "source": "denue",
            "borough_id": None,
            "listed_since": datetime.now(UTC).date(),
            "kind": "unclassified",
        }


class AuctionLot(BaseModel):
    """A lot ranked by the Cup of Excellence jury in Mexico, before its auction."""

    score: float = Field(ge=80, le=100, description="The jury's score, e.g. 89.5")
    state: str | None = Field(
        None,
        description="The Mexican state it was grown in, e.g. Veracruz, Chiapas, Oaxaca, Puebla",
    )
    processing_method: str | None = Field(None, description=METHODS)
    varieties: list[str] = Field(
        [], description="Lower case, e.g. ['gesha'], ['bourbon', 'typica'], ['marsellesa']"
    )
    weight_kg: float | None = Field(None, gt=0, le=10_000, description="The lot's weight, kg")
    national_winner: bool = Field(
        False, description="Scored under the CoE's line and sold in the national winners' auction"
    )
    year: int | None = Field(
        None, ge=2012, le=2100, description="The auction's year; defaults to this year"
    )

    _lowered = field_validator("processing_method", mode="before")(_lower)

    def to_item(self) -> dict[str, Any]:
        """The lot as a row of `cup_of_excellence`, its price not known yet."""
        return self.model_dump(exclude={"year"}) | {
            "lot_id": "request",
            "year": self.year or datetime.now(UTC).year,
            "varieties": [v.strip().lower() for v in self.varieties],
            "price_usd_per_lb": None,
        }


class Household(BaseModel):
    """A Mexican household, as INEGI's income and expenditure survey would describe it."""

    state: str = Field(
        description="The Mexican state, as INEGI names it, e.g. Ciudad de México, Veracruz"
    )
    members: int = Field(ge=1, le=30, description="How many people live in it")
    income_month_mxn: float = Field(ge=0, description="The household's income in a month, pesos")
    grows_coffee: bool = Field(False, description="Takes coffee from its own harvest")

    def to_item(self) -> dict[str, Any]:
        """The household as a row of `household_coffee`: income kept per quarter, as the
        survey keeps it, and what it bought not known."""
        return {
            "household_id": "request",
            "year": datetime.now(UTC).year,  # the item's time only: no feature reads it
            "state": self.state,
            "members": self.members,
            "income_quarter_mxn": self.income_month_mxn * 3,
            "own_harvest_quarter_mxn": 1.0 if self.grows_coffee else 0.0,
            "instant_quarter_mxn": None,
            "ground_quarter_mxn": None,
            "prepared_quarter_mxn": None,
        }
