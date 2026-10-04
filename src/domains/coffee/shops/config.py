"""A coffee shop's config: the shop, and how its export is simulated.

Everything a coffee shop is - its menu and prices, recipes, opening hours, staff and
wages, how many tickets a day, how long a drink takes - is data, in its file under
`businesses/`, so a demo shop is changed by editing a file and a real one would be
described the same way. What every shop shares - sources, simulation, models - is
`config.yaml`. What each number rests on is said beside it: a real source it is anchored
to, or an assumption to replace with the owner's own.
"""

from datetime import date, datetime, time, timedelta
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
    """A shop: what it is, where, what it sells and how - and what its owner aims at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    concept: str
    zone_id: str  # the urban AGEB it opens in (its file says why this one)
    latitude: float = Field(ge=-90, le=90)  # where it stands: its neighbours are counted from here
    longitude: float = Field(ge=-180, le=180)
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
    # The owner's aims, which the studies judge the shop against: the gross margin each
    # category of the menu should keep (price less what goes into it, % of the price), and
    # the minutes from order to ready a customer should not wait past.
    target_margins: dict[str, float]
    wait_target_minutes: float = Field(gt=0)

    @model_validator(mode="after")
    def _every_category_has_a_target(self) -> Self:
        missing = sorted({item.category for item in self.menu} - set(self.target_margins))
        if missing:
            raise ValueError(f"No target margin for the menu's categories {missing}")
        bad = sorted(c for c, pct in self.target_margins.items() if not 0 < pct < 100)
        if bad:
            raise ValueError(f"A target margin is a percent between 0 and 100: {bad}")
        return self

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

    @model_validator(mode="after")
    def _someone_is_on_shift_whenever_it_is_open(self) -> Self:
        """Checked every quarter of an hour: an open shop with nobody behind the counter
        would serve its customers anyway in the simulation, and pay nobody for it."""
        day = date(2000, 1, 3)  # a Monday: only the time of day matters
        uncovered = []
        for weekday, opens in sorted(self.opens.items()):
            moment = datetime.combine(day, opens)
            while moment.time() < self.closes[weekday]:
                if not any(weekday in s.days and s.starts <= moment.time() < s.ends
                           for s in self.shifts):  # fmt: skip
                    uncovered.append(f"{weekday} {moment:%H:%M}")
                moment += timedelta(minutes=15)
        if uncovered:
            raise ValueError(f"Open with nobody on shift (ISO weekday, time): {uncovered[:5]}")
        return self


class SimulationConfig(BaseModel):
    """Where the simulation reads its real anchors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sales_pattern: str  # the file source whose real sales give hours, days and elasticity
    green_coffee: str  # the parent's: green coffee in pesos, monthly
    inflation: str  # the parent's: the consumer price index, monthly
    green_indicator: str  # which green coffee price the beans follow


class StudiesConfig(BaseModel):
    """How every shop's studies are run (`analysis.py`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Price changes to weigh, % on every price of the menu: what each would do to the day.
    scenarios: list[float] = Field(min_length=1)
    baseline_weeks: int = Field(ge=1)  # "now": the last weeks a scenario starts from
    outlook_months: int = Field(ge=1)  # how far ahead a price alert looks at costs
    menu_months: int = Field(ge=1)  # the months menu engineering reads
    radii_m: list[float] = Field(min_length=1)  # coffee shops counted within each of these
    resamples: int = Field(ge=100)  # of whole weeks, for the price response's interval
    seed: int
    # The parent's tables each study reads, by the names the shop reads them by.
    green_outlook: str  # where green coffee may be in some months, in a range
    coffee_shops: str
    zones: str  # one row per urban AGEB: residents, density, schooling, internet
    zone_expectations: str  # how many coffee shops an AGEB like each would have
    roaster_offers: str  # what the city's roasters charge for a kilogram


class CoffeeShopConfig(DomainConfig):
    """One shop: a subdomain of coffee, reading the coffee tables listed in `parent`."""

    shop: ShopConfig
    simulation: SimulationConfig
    studies: StudiesConfig

    @model_validator(mode="after")
    def _simulation_reads_what_the_shop_lists(self) -> Self:
        """Its real sales are a source of the shop's; the market is its parent's tables,
        each listed in `parent` - the simulation reads nothing else."""
        if self.simulation.sales_pattern not in self.sources:
            raise ValueError(f"No source '{self.simulation.sales_pattern}' to anchor the sales to")
        for name in (self.simulation.green_coffee, self.simulation.inflation):
            if name not in self.parent_tables:
                raise ValueError(f"The simulation reads {name}, which `parent` does not list")
        studies = self.studies
        for name in (studies.green_outlook, studies.coffee_shops, studies.zones,
                     studies.zone_expectations, studies.roaster_offers):  # fmt: skip
            if name not in self.parent_tables:
                raise ValueError(f"The studies read {name}, which `parent` does not list")
        return self
