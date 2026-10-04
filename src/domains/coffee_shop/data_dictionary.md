# Data dictionary

The coffee shop's tables: one neighbourhood specialty coffee shop in Mexico City, from
1 January 2025 to 31 August 2026. **The shop does not exist**: its point-of-sale export is
simulated (`domains/coffee_shop/simulate.py`), anchored to real data - when people buy and
how they answer a price come from a real coffee machine's sales, what the shop pays
follows green coffee in pesos and Mexico's consumer price index. Menu, recipes, hours,
staff and wages are assumptions in `config.yaml`. Money is in Mexican pesos of the day.

Every table below is an immutable Parquet partition and a DuckDB view
(`SELECT * FROM clean.sales`). The Pandera contract in the code is the authority.

## `clean.sales` — one line of a ticket

A product sold, at the price it was sold at. 114,297 rows.

| Column | Type | Meaning |
|---|---|---|
| `ticket_id` | String | `<yyyymmdd>-<n>`: the ticket, numbered within its day |
| `line` | Int | The line within the ticket, from 1 |
| `sold_at` | Datetime | When the ticket was rung up |
| `date`, `hour` | Date, Int | Its day, and the hour it started in (0-23) |
| `weekday` | Int | ISO weekday: 1 is Monday, 7 Sunday |
| `product` | String | `espresso`, `americano`, `cortado`, `cappuccino`, `latte`, `filter_v60`, `cold_brew`, `hot_chocolate`, `croissant`, `pan_dulce` |
| `category` | String | `espresso`, `milk`, `filter`, `chocolate`, `pastry` |
| `quantity` | Int | Units on the line |
| `unit_price_mxn` | Float | The menu price that day, pesos |
| `line_total_mxn` | Float | `unit_price_mxn` × `quantity` |
| `channel` | String | `dine_in` or `takeaway` |
| `payment` | String | `card` or `cash` |

## `clean.orders` — one ticket, from order to ready

77,345 rows. A customer who would have waited longer than their patience left without
buying and is in no table: the export never sees a lost sale.

| Column | Type | Meaning |
|---|---|---|
| `ticket_id` | String | Joins `clean.sales` |
| `ordered_at`, `ready_at` | Datetime | When it was ordered, and when its last item was ready |
| `date`, `hour` | Date, Int | Of `ordered_at` |
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
| `ingredient` | String | `coffee_beans`, `milk`, `chocolate`, `cup`, `filter_paper`, `croissant`, `pan_dulce` |
| `quantity` | Float | In the ingredient's unit, for one |
| `unit` | String | `g`, `ml` or `piece` |

## `clean.purchases` — one ingredient bought for a week

Bought on Mondays: the week's use and its waste, at that month's cost. Coffee beans
move with half the green coffee price in pesos; the rest moves with the consumer
price index.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The Monday it was bought |
| `ingredient`, `unit` | String | As in `clean.recipes` |
| `quantity` | Float | Units bought |
| `cost_mxn` | Float | What it cost, pesos |
| `unit_cost_mxn` | Float | `cost_mxn` ÷ `quantity` |

## `clean.shifts` — one shift worked

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
nothing is a row. 8,251 rows. The items of `hourly_demand`.

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
(`domains/coffee_shop/features.py`).

| Column | Type | Meaning |
|---|---|---|
| `hour_id`, `month`, `date` | | Keys back to `clean.shop_hours` |
| `hour`, `weekday` | Int | As in `clean.shop_hours` |
| `weekday_hour` | String | `<weekday>-<hh>`: the slot, `1-08` is a Monday at eight |
| `price_level` | Float | The menu's that day; a request may move it by a percent |
| `tickets` | Float | Target |

## `predictions.hourly_demand_predictions` — expected tickets an hour

| Column | Type | Meaning |
|---|---|---|
| `hour_id`, `month`, `date` | | Keys back to `clean.shop_hours` |
| `prediction` | Float | Expected `tickets` for an hour like it |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## Raw layer

`raw/<source>/ingested_at=<timestamp>/` holds each ingestion exactly as it came, next to
a manifest with its sha256. `vending_sales` is the real coffee machine's CSV; each
`pos_*` source is one table of the simulated export, as JSON - its rows, and the anchors
it was simulated from. The same anchors and config make the same bytes, so running the
simulation again stores nothing new.
