"""What the prediction API accepts for coffee: a lot, before it is cupped."""

from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel, Field


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

    def to_item(self) -> dict[str, Any]:
        """The lot as a row of the item table: `graded_on` is its time column."""
        return self.model_dump(exclude={"graded_on"}) | {
            "grading_date": self.graded_on or datetime.now(UTC).date()
        }
