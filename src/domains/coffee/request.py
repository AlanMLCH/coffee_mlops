"""What the prediction API accepts for coffee: a lot before it is cupped, a bag on a
shelf, and a month whose green coffee price is not out yet."""

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
