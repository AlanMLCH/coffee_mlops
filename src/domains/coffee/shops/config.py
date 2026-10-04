"""A coffee shop's config: the shop, and how its export is simulated.

Everything a coffee shop is - its menu and prices, recipes, opening hours, staff and
wages, how many tickets a day, how long a drink takes - is data, in its file under
`businesses/`, so a demo shop is changed by editing a file and a real one would be
described the same way. What every shop shares - sources, simulation, models - is
`config.yaml`. What each number rests on is said beside it: a real source it is anchored
to, or an assumption to replace with the owner's own.
"""

from datetime import date, time
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlops_core.config import DomainConfig


class MenuItem(BaseModel):
    """A product on the menu: its price at opening, how much of the sales it makes, how
    long it takes, and what goes into one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    product: str
    category: str  # espresso, milk, filter, chocolate, pastry
    price_mxn: float = Field(gt=0)
    share: float = Field(gt=0)  # of the items sold, before prices move
    prep_minutes: float = Field(ge=0)  # a barista's hands on it, without a queue
    recipe: dict[str, float]  # ingredient -> quantity in its unit


class Ingredient(BaseModel):
    """What the shop buys: its unit, its cost when the shop opens, and what its cost
    follows from then on - the green coffee price in pesos, the consumer price index, or
    nothing (a fixed price)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str  # g, ml, piece
    cost_mxn: float = Field(gt=0)  # per unit, in the month the shop opens
    follows: Literal["green_coffee", "inflation"] | None = None  # None: a fixed price
    # The share of its cost that moves with what it follows: roasted beans cost the green
    # coffee in them, and a roaster's work and margin, which do not move with it.
    pass_through: float = Field(1.0, ge=0, le=1)
    waste_pct: float = Field(0, ge=0, lt=100)  # bought and thrown away


class PriceChange(BaseModel):
    """A change to every price on the menu, from a day on - for good, or until `until`
    (a promotion)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    starts: date
    until: date | None = None
    change_pct: float


class Shift(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    starts: time
    ends: time
    days: list[int]  # ISO weekdays it is worked: 1 is Monday


class ShopConfig(BaseModel):
    """The demo shop: a neighbourhood specialty coffee shop that does not exist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    concept: str
    zone_id: str  # the urban AGEB it opens in (see config.yaml for why this one)
    first_day: date
    last_day: date
    opens: dict[int, time]  # ISO weekday -> opening time
    closes: dict[int, time]
    tickets_per_day: float = Field(gt=0)  # an ordinary weekday at opening prices
    items_per_ticket: list[float] = Field(min_length=1)  # P(1 item), P(2 items), ...
    dine_in_share: float = Field(ge=0, le=1)  # of the tickets; the rest are taken away
    card_share: float = Field(ge=0, le=1)  # of the tickets; the rest are paid in cash
    menu: list[MenuItem] = Field(min_length=1)
    ingredients: dict[str, Ingredient]
    price_changes: list[PriceChange] = []
    shifts: list[Shift] = Field(min_length=1)
    hourly_wage_mxn: float = Field(gt=0)
    seed: int

    @model_validator(mode="after")
    def _recipes_use_known_ingredients(self) -> Self:
        unknown = sorted(
            {name for item in self.menu for name in item.recipe} - set(self.ingredients)
        )
        if unknown:
            raise ValueError(f"Recipes use ingredients the shop does not buy: {unknown}")
        if abs(sum(self.items_per_ticket) - 1) > 1e-6:
            raise ValueError("items_per_ticket must add up to 1")
        if set(self.opens) != set(self.closes):
            raise ValueError("Every day the shop opens it also closes")
        return self


class SimulationConfig(BaseModel):
    """Where the simulation reads its real anchors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sales_pattern: str  # the file source whose real sales give hours, days and elasticity
    green_coffee: str  # the parent's: green coffee in pesos, monthly
    inflation: str  # the parent's: the consumer price index, monthly
    green_indicator: str  # which green coffee price the beans follow


class CoffeeShopConfig(DomainConfig):
    """One shop: a subdomain of coffee, reading the coffee tables listed in `parent`."""

    shop: ShopConfig
    simulation: SimulationConfig

    @model_validator(mode="after")
    def _simulation_reads_what_the_shop_lists(self) -> Self:
        """Its real sales are a source of the shop's; the market is its parent's tables,
        each listed in `parent` - the simulation reads nothing else."""
        if self.simulation.sales_pattern not in self.sources:
            raise ValueError(f"No source '{self.simulation.sales_pattern}' to anchor the sales to")
        for name in (self.simulation.green_coffee, self.simulation.inflation):
            if name not in self.parent_tables:
                raise ValueError(f"The simulation reads {name}, which `parent` does not list")
        return self
