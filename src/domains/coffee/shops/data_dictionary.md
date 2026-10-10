# Data dictionary

A coffee shop's tables: every shop of the coffee domain has these, each its own and
no other shop's. **The demo shops do not exist**: their point-of-sale exports are simulated
(`domains/coffee/shops/simulate.py`), anchored to real data - when people buy and how they
answer a price come from a real coffee machine's sales, what a shop pays follows green
coffee in pesos and Mexico's consumer price index. Menu, recipes, hours, staff and wages
are assumptions in the shop's file. Money is in Mexican pesos of the day.

Every table below is an immutable Parquet partition and a DuckDB view
(`SELECT * FROM clean.sales`). The Pandera contract in the code is the authority.

**What no table holds.** No other coffee shop's menu or prices: `coffee.clean.coffee_shops`
says where they are, not what they charge. And no customer: a point-of-sale export
carries none, so nothing follows a person from one ticket to the next. A date column
holds days: a period longer than a day is a range of them.

## `clean.sales` — one line of a ticket

A product sold, at the price it was sold at. **A row is a line of a ticket, not a
ticket**: a ticket of three products is three rows, its channel and payment on each.

| Column | Type | Meaning |
|---|---|---|
| `ticket_id` | String | `<yyyymmdd>-<n>`: the ticket, numbered within its day |
| `line` | Int | The line within the ticket, from 1 |
| `sold_at` | Datetime | When the ticket was rung up |
| `date`, `hour` | Date, Int | Its day, and the hour it started in (0-23) |
| `weekday` | Int | ISO weekday: 1 is Monday, 7 Sunday |
| `product` | String | As the shop's menu names it, lower case with `_` (`americano`, `latte`, `cafe_de_olla`): `SELECT DISTINCT product FROM clean.menu_prices` lists them |
| `category` | String | `espresso`, `milk`, `filter`, `chocolate` or `pastry` |
| `quantity` | Int | Units on the line |
| `unit_price_mxn` | Float | The menu price that day, pesos |
| `line_total_mxn` | Float | `unit_price_mxn` × `quantity` |
| `channel` | String | `dine_in` or `takeaway`, the same on every line of a ticket |
| `payment` | String | `card` or `cash`, the same on every line of a ticket |

## `clean.orders` — one ticket, from order to ready

A customer who would have waited longer than their patience left without buying and is in
no table: the export never sees a lost sale.

| Column | Type | Meaning |
|---|---|---|
| `ticket_id` | String | Joins `clean.sales` |
| `ordered_at`, `ready_at` | Datetime | When it was ordered, and when its last item was ready |
| `date`, `month`, `hour` | Date, String, Int | Of `ordered_at`; `month` is `YYYY-MM` |
| `weekday` | Int | ISO weekday: 1 is Monday |
| `minutes` | Float | From order to ready: the queue and the making |
| `items` | Int | Items on the ticket |
| `baristas` | Int | Staff on shift when it was ordered |

## `clean.menu_prices` — a product's price over a period

The menu's history: a new period wherever a price change starts or a promotion ends.

| Column | Type | Meaning |
|---|---|---|
| `product`, `category` | String | As in `clean.sales` |
| `price_mxn` | Float | The price, pesos |
| `valid_from` | Date | First day of the period |
| `valid_to` | Date? | Last day of it; null for the prices in force |

## `clean.recipes` — what goes into one of a product

| Column | Type | Meaning |
|---|---|---|
| `product` | String | As in `clean.sales` |
| `ingredient` | String | As the shop's file names it (`coffee_beans`, `milk`, `cup`, ...) |
| `quantity` | Float | In the ingredient's unit, for one |
| `unit` | String | `g`, `ml` or `piece` |

## `clean.purchases` — one ingredient bought for a week

Bought on Mondays: the week's use and its waste, at that month's cost. Coffee beans
move in part with the green coffee price in pesos (the share the shop's file says); the
rest moves with the consumer price index.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The Monday it was bought |
| `ingredient`, `unit` | String | As in `clean.recipes` |
| `quantity` | Float | Units bought |
| `cost_mxn` | Float | What it cost, pesos |
| `unit_cost_mxn` | Float | `cost_mxn` ÷ `quantity` |

## `clean.shifts` — one shift worked

A row is one person's shift, and carries what it cost.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The day |
| `staff_id` | String | `<role>-<n>`: the same turn every week |
| `role` | String | `barista` or `helper` |
| `starts_at`, `ends_at` | Datetime | The shift |
| `hours` | Float | Its length |
| `hourly_wage_mxn` | Float | What an hour costs the shop that month, wage and employer's share, pesos |
| `cost_mxn` | Float | `hourly_wage_mxn` × `hours` |

## `clean.shop_hours` — one hour the shop was open

Every open hour from the first sale to the last, **zeros included**: an hour that sold
nothing is a row. The items of `hourly_demand`. **A row is an hour, not a day**: its
columns count that hour alone.

| Column | Type | Meaning |
|---|---|---|
| `hour_id` | String | `<yyyy-mm-dd>T<hh>` |
| `date`, `month` | Date, String | The day, and `YYYY-MM` |
| `hour` | Int | The hour it starts at, 0-23 |
| `weekday` | Int | ISO weekday: 1 is Monday |
| `tickets` | Float | Tickets rung up in the hour |
| `items` | Float | Units sold in it |
| `revenue_mxn` | Float | What they came to, pesos |
| `price_level` | Float | The day's menu against its first prices: 1.06 is 6% above opening |
| `baristas` | Int | Staff on shift at the half hour |
| `mean_minutes` | Float? | Mean minutes from order to ready; null in an hour with no ticket |

## `features.hourly_demand_features` — tickets an hour, model input

One row of `clean.shop_hours` with what the hour may know before it happens
(`domains/coffee/shops/features.py`).

| Column | Type | Meaning |
|---|---|---|
| `hour_id`, `month`, `date` | | Keys back to `clean.shop_hours` |
| `hour`, `weekday` | Int | As in `clean.shop_hours` |
| `weekday_hour` | String | `<weekday>-<hh>`: the slot, `1-08` is a Monday at eight |
| `price_level` | Float | The menu's that day; a request may move it by a percent |
| `slot_last` | Float? | Tickets the same weekday and hour brought the last time before the item's day (last week's) |
| `slot_mean_4w` | Float? | Their mean over its last four times |
| `history_price_level` | Float? | The menu's level over those four times: a price changed since is the difference |
| `tickets` | Float | Target |

## `predictions.hourly_demand_predictions` — expected tickets an hour

| Column | Type | Meaning |
|---|---|---|
| `hour_id`, `month`, `date` | | Keys back to `clean.shop_hours` |
| `prediction` | Float | Expected `tickets` for an hour like it |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.order_minutes_features` — minutes an order takes, model input

One row of `clean.orders` with what the order may know before it is made
(`domains/coffee/shops/features.py`).

| Column | Type | Meaning |
|---|---|---|
| `ticket_id`, `month`, `date` | | Keys back to `clean.orders` |
| `hour`, `weekday`, `weekday_hour` | | Its slot, as in `clean.shop_hours` |
| `items` | Int | Items on the ticket |
| `baristas` | Int | Staff on shift when it was ordered |
| `tickets_in_hour` | Float | Tickets its hour brought: how busy the counter was. A request gives it |
| `minutes` | Float | Target: from order to ready |

## `predictions.order_minutes_predictions` — expected minutes an order takes

| Column | Type | Meaning |
|---|---|---|
| `ticket_id`, `month`, `date` | | Keys back to `clean.orders` |
| `prediction` | Float | Expected `minutes` for an order like it |
| `lower`, `upper` | Float | The range it falls in four times in five |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `analysis.profile` — the shop itself, one row

| Column | Type | Meaning |
|---|---|---|
| `name`, `concept` | String | What the shop is called and what it is |
| `zone_id` | String | The urban AGEB it stands in (`coffee.clean.census_zones`) |
| `latitude`, `longitude` | Float | Where it stands: the AGEB's centroid |
| `first_day`, `last_day` | Date | The history its tables hold |
| `products` | Int | Products on its menu |
| `wait_target_minutes` | Float | Minutes from order to ready the owner wants no customer to wait past |

## `analysis.product_margins` — what one product earned in one month

What went into one is its recipe at the month's purchase prices, times what the shop buys
for each unit it uses: waste is part of the cost.

| Column | Type | Meaning |
|---|---|---|
| `month` | Date | First day of the month |
| `product`, `category` | String | As in `clean.sales` |
| `units` | Int | Units sold that month |
| `price_mxn` | Float | What one sold for on average that month, promotions included |
| `unit_cost_mxn` | Float | What went into one |
| `margin_mxn` | Float | `price_mxn` - `unit_cost_mxn`: the gross margin of one |
| `margin_pct` | Float | The same, % of the price |
| `beans_share_pct` | Float | Coffee's share of `unit_cost_mxn`, %; 0 for what holds none |

## `analysis.price_response` — how the shop's customers answer a price

The elasticity of a day's tickets to the menu's price level, from the shop's own price
changes, with weekday effects; the interval resamples whole weeks. Every price moved
together, so it is the menu's, not a product's. **Use the `whole day` row** to weigh a
price change.

| Column | Type | Meaning |
|---|---|---|
| `scope` | String | `whole day`, or `quiet hours` (slots whose orders seldom wait past the target: the queue did not turn customers away) |
| `elasticity` | Float? | % change in tickets for a 1% change in every price; -1.2 means a 10% rise loses about 11% of the tickets. Null with one price all along |
| `elasticity_low`, `elasticity_high` | Float? | Its 95% interval |
| `days` | Int | Days it was read from |
| `price_levels` | Int | Distinct menu levels the history holds |
| `level_min`, `level_max` | Float | The lowest and highest level the menu had (1 = opening prices) |

## `analysis.price_scenarios` — a day now, and with every price moved

"Now" is the last eight weeks. Tickets move by the whole day's elasticity, taken as
constant - beyond the levels the menu has had (`within_history` false) an assumption, not
evidence. Gross profit is revenue less what went into what was sold, before staff and rent.

| Column | Type | Meaning |
|---|---|---|
| `change_pct` | Float | Every price moved by this %; 0 is now |
| `price_level` | Float | The menu's level it would make (1 = opening prices) |
| `within_history` | Boolean | A level the menu has had |
| `tickets_per_day`, `revenue_per_day` | Float | A day's tickets and revenue, pesos |
| `gross_profit_per_day` | Float | Pesos a day |
| `gross_profit_low`, `gross_profit_high` | Float | The same at either end of the elasticity's interval |
| `versus_now_pct`, `versus_now_low_pct`, `versus_now_high_pct` | Float | Gross profit against now, %: **the answer to "should I raise my prices?"** - a range that straddles 0 means the shop's history cannot tell |

## `analysis.price_alerts` — when to raise: each product against the margin it should keep

The latest month, against the gross margin the owner aims at for its category - now, and at
the horizon if green coffee reaches the top of the coffee domain's outlook range and prices
in general keep rising as in the last year.

| Column | Type | Meaning |
|---|---|---|
| `product`, `category`, `month` | | The product and the month read |
| `price_mxn`, `unit_cost_mxn`, `margin_pct` | Float | As in `analysis.product_margins` |
| `target_pct` | Float | The margin its category should keep, % (the shop's file) |
| `raise_needed_pct` | Float | Every price of it up this % restores the target now; 0 when it holds |
| `horizon_months` | Int | How far ahead the alert looks |
| `green_coffee_high_pct` | Float? | The top of green coffee's range at the horizon, % from now |
| `inflation_pct` | Float? | The consumer price index's last-year pace, over the horizon, % |
| `margin_at_horizon_pct` | Float? | The margin then, at today's price |
| `status` | String | `raise` (below target now), `watch` (would fall below at the horizon) or `ok` |

## `analysis.inflation_impact` — what inflation did to the shop, month by month

Indexes against the shop's first month (1 = as it opened); `real_` deflates by Mexico's
consumer price index.

| Column | Type | Meaning |
|---|---|---|
| `month` | Date | First day of the month |
| `price_level`, `real_price_level` | Float | The menu's level, in pesos of the day and of the first month: a real level that falls is a rise inflation took back |
| `cost_index`, `real_cost_index` | Float | What the month's mix of products cost to make |
| `wage_index` | Float | An hour of staff |
| `cpi_index` | Float? | The consumer price index against the first month |
| `gross_margin_pct` | Float | Revenue less what went into it, % of revenue |

## `analysis.menu_engineering` — which products carry the menu

The last three months. Popular: at least 70% of an equal share of the items. Profitable: a
margin in pesos at least the menu's, weighted by what sells.

| Column | Type | Meaning |
|---|---|---|
| `product`, `category` | String | As in `clean.sales` |
| `units`, `revenue_mxn` | | Sold in the window |
| `margin_mxn` | Float | Gross margin of one, pesos |
| `mix_pct` | Float | Its share of the items sold, % |
| `class` | String | `star` (popular, profitable), `plowhorse` (popular, thin), `puzzle` (profitable, slow), `dog` (neither) |
| `margin_vs_menu_mxn` | Float | Its margin against the menu's, pesos |

## `analysis.staffing` — each weekday-and-hour slot: its crowd, its hands, its waits

| Column | Type | Meaning |
|---|---|---|
| `weekday`, `hour` | Int | ISO weekday (1 is Monday) and the hour it starts at |
| `tickets` | Float | Tickets in the slot on an average day |
| `staff` | Float | Staff on shift, on average |
| `tickets_per_staff_hour` | Float? | Tickets for each pair of hands; null with nobody on shift |
| `staff_cost_per_ticket_mxn` | Float? | What staff costs for each ticket of the slot, pesos; null when it sold nothing |
| `wait_p90_minutes` | Float? | Nine orders in ten are ready within this, order to ready |
| `over_target_pct` | Float? | Orders that took longer than the owner's target, % |
| `long_waits` | Boolean | `wait_p90_minutes` past the target: too few hands for the crowd |

## `analysis.neighbourhood` — the shop's corner of the city against the rest

From the coffee domain's tables. `borough` and `city` are the median AGEB's; a measure
whose table is not built is left out. There is no public source of what coffee shops charge
for a drink in Mexico City: the shop's prices are compared with nobody's.

| Column | Type | Meaning |
|---|---|---|
| `measure` | String | `coffee shops within <r> m` (DENUE's and OSM's coffee shops, each place once: one both list is counted by DENUE's row), `residents`, `density`, `schooling`, `homes with internet`, `coffee shops listed in the AGEB`, `coffee shops an AGEB like it would have` (the `zones` model, held out) |
| `shop` | Float | The shop's, or its AGEB's |
| `borough`, `city` | Float? | The median AGEB of its borough, of the city; null for the counts within a radius |
| `unit` | String | What the numbers count |

## `analysis.monthly_results` — what was left each month, after every cost

One row a month: what was sold and every cost against it. Rent and services are the
shop's file's assumptions (from listing portals' asking rents and small coffee shops'
guides, 2026), moved with the consumer price index; the card fee is a terminal's
commission and its VAT on the card sales. **The answer to "how much did I make" and
"what was left"**: `operating_profit_mxn`, before taxes, the owner's own pay and the
equipment's wear.

| Column | Type | Meaning |
|---|---|---|
| `month` | Date | First day of the month |
| `revenue_mxn` | Float | What was sold, pesos |
| `ingredients_mxn` | Float | What went into what was sold, waste included |
| `card_fees_mxn` | Float | What the card terminal kept |
| `staff_mxn` | Float | Every shift's cost, the employer's share included |
| `rent_mxn`, `services_mxn` | Float | The premises (rent and maintenance), and electricity, gas, water and internet |
| `operating_profit_mxn` | Float | `revenue_mxn` less every cost above |
| `operating_margin_pct` | Float | The same, % of revenue |
| `prime_cost_pct` | Float | Ingredients and staff together, % of revenue: what owners read first |

## `analysis.sales_outlook` — the coming months' sales, at today's demand and prices

**The answer to "how much will I sell" next month or in a month ahead.** The next twelve
months after the last one the tables hold: each weekday's tickets and sales on an average
day of the last eight weeks, times the days the shop opens in the month. Its range comes
from its own misses on months that had already happened (`analysis.sales_backtest`). It
holds today's prices and crowd: a price change, a competitor or a season the history does
not hold is not in it.

| Column | Type | Meaning |
|---|---|---|
| `month` | Date | First day of the month forecast |
| `horizon_months` | Int | Months after the last one the tables hold: 1 is next month |
| `days_open` | Int | Days the shop opens that month |
| `tickets` | Float | Tickets the month would bring |
| `revenue_mxn` | Float | What it would sell, pesos |
| `revenue_low_mxn`, `revenue_high_mxn` | Float? | Its range: one month in ten fell below, one in ten above, by the forecast's past misses |

## `analysis.sales_backtest` — how the outlook did on months that had happened

At every past month end with eight weeks behind it, the same forecast for the next one to
three months, set against what they sold - and against the month just closed, repeated
(`naive_mxn`): the forecast earns its place by missing less.

| Column | Type | Meaning |
|---|---|---|
| `origin` | Date | The last month the forecast knew |
| `month` | Date | The month forecast |
| `horizon_months` | Int | Months after `origin` |
| `forecast_mxn`, `actual_mxn` | Float | What it said, and what was sold |
| `error_pct` | Float | `forecast_mxn` against `actual_mxn`, %; above zero is too high |
| `naive_mxn`, `naive_error_pct` | Float | The origin month's sales, repeated, and how far that was |

## Raw layer

`raw/<source>/ingested_at=<timestamp>/` holds each ingestion exactly as it came, next to
a manifest with its sha256. `vending_sales` is the real coffee machine's CSV; each
`pos_*` source is one table of the simulated export, as JSON - its rows, and the anchors
it was simulated from. The same anchors and config make the same bytes, so running the
simulation again stores nothing new.
