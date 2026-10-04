"""What a coffee shop's prediction API accepts: an hour of a day, or an order in it - and
what the owner would change before it comes."""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class ShopHour(BaseModel):
    """An hour the shop is open, to know how many tickets it brings - at today's menu
    prices, or with every price moved."""

    day: date = Field(description="The day, YYYY-MM-DD")
    hour: int = Field(ge=0, le=23, description="The hour it starts at, 0-23: 8 is 08:00-08:59")
    price_change_pct: float = Field(
        0.0,
        ge=-50,
        le=100,
        description="Every menu price moved by this percent from the menu in force that "
        "day; 0 keeps the prices as they are",
    )

    def to_item(self) -> dict[str, Any]:
        """The hour as a row of `shop_hours`, its sales not known yet."""
        return {
            "hour_id": f"{self.day.isoformat()}T{self.hour:02d}",
            "date": self.day,
            "month": self.day.strftime("%Y-%m"),
            "hour": self.hour,
            "weekday": self.day.isoweekday(),
            "price_change_pct": self.price_change_pct,
        }


class ShopOrder(BaseModel):
    """An order, to know how long it would take from order to ready - with the hands and
    the crowd the owner expects in its hour."""

    day: date = Field(description="The day, YYYY-MM-DD")
    hour: int = Field(ge=0, le=23, description="The hour it is ordered in, 0-23")
    items: int = Field(1, ge=1, le=10, description="Items on the ticket")
    baristas: int = Field(ge=1, le=6, description="Staff on shift behind the counter")
    tickets_in_hour: float = Field(
        ge=0, le=200, description="Tickets the whole hour brings; hourly_demand predicts it"
    )

    def to_item(self) -> dict[str, Any]:
        """The order as a row of `orders`, its minutes not known yet."""
        return {
            "date": self.day,
            "month": self.day.strftime("%Y-%m"),
            "hour": self.hour,
            "weekday": self.day.isoweekday(),
            "items": self.items,
            "baristas": self.baristas,
            "tickets_in_hour": self.tickets_in_hour,
        }
