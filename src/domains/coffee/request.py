"""What the prediction API accepts for coffee: a lot before it is cupped, a bag on a shelf."""

from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _lower(value: object) -> object:
    """A closed vocabulary is lower case ("gesha", "washed", "almanegra"); "Gesha" is the
    same word, and the model would otherwise take it for a category it never saw."""
    return value.strip().lower() if isinstance(value, str) else value


class Lot(BaseModel):
    """What a caller knows about a green coffee lot before it is cupped."""

    country: str
    variety: str | None = None
    processing_method: str | None = None
    color: str | None = None
    altitude_m: float | None = Field(None, ge=0, le=9000)
    moisture_pct: float | None = Field(None, ge=0, le=100)
    category_one_defects: int = Field(0, ge=0)
    category_two_defects: int = Field(0, ge=0)
    quakers: int | None = Field(None, ge=0)
    graded_on: date | None = Field(None, description="Defaults to today (UTC).")

    _lowered = field_validator("variety", "processing_method", "color", mode="before")(_lower)

    def to_item(self) -> dict[str, Any]:
        """The lot as a row of the item table: `graded_on` is its time column."""
        return self.model_dump(exclude={"graded_on"}) | {
            "grading_date": self.graded_on or datetime.now(UTC).date()
        }


class Offer(BaseModel):
    """A bag of roasted coffee as a shop would list it, to price it per kilogram."""

    shop: str = Field(description="The roaster, as the catalogues name it, e.g. almanegra")
    bag_grams: float = Field(gt=0, le=20_000)
    country: str | None = Field(None, description="As PSD names it, e.g. Mexico")
    state: str | None = Field(None, description="As SIAP names it, e.g. Oaxaca")
    processing_method: str | None = Field(None, description="washed, natural, honey, ...")
    variety: str | None = Field(None, description="As the CQI spells it, e.g. gesha")
    producer: str | None = None
    altitude_m: float | None = Field(None, ge=0, le=9000)
    observed_on: date | None = Field(None, description="Defaults to today (UTC).")

    _lowered = field_validator("shop", "processing_method", "variety", mode="before")(_lower)

    def to_item(self) -> dict[str, Any]:
        """The bag as a row of the offers table: `observed_on` is its time column."""
        return self.model_dump(exclude={"observed_on"}) | {
            "observed_on": self.observed_on or datetime.now(UTC).date()
        }
