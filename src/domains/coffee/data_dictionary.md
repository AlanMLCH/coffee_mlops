# Data dictionary

Every table below is an immutable Parquet partition and a DuckDB view
(`SELECT * FROM clean.coffee_reviews`). The Pandera contract in the code is the
authority; a test fails if a column here goes missing from it.

## `clean.coffee_reviews` — one graded lot

Both CQI snapshots harmonised into one table. 1,516 rows.

| Column | Type | Meaning |
|---|---|---|
| `review_id` | String | `<snapshot>-<upstream id>`, unique |
| `snapshot` | String | `cqi_2018` or `cqi_2023`: which scrape the row came from |
| `country` | String | Origin, renamed to match USDA PSD (Hawaii and Puerto Rico → United States): `Ethiopia`, `Colombia` |
| `region` | String? | Growing region, free text as submitted |
| `variety` | String? | Lower-cased (`caturra`, `gesha`, `sl28`, `ethiopian heirlooms`); blends and "unknown" are kept as written |
| `processing_method` | String? | `washed`, `natural`, `honey`, `semi_washed`, `other` |
| `color` | String? | Green bean colour, normalised to a closed vocabulary |
| `grading_date` | Date | When the lot was cupped. Drives the temporal split and the market-context join |
| `altitude_m` | Float? | Metres. Outside 300-3,000 m it is treated as unrecorded |
| `moisture_pct` | Float? | Percentage (the 2018 scrape stored a fraction). 0 means not measured |
| `category_one_defects` | Int | Primary (visual) green defects |
| `category_two_defects` | Int | Secondary defects |
| `quakers` | Int? | Unripe beans that fail to roast |
| `aroma`, `flavor`, `aftertaste`, `acidity`, `body`, `balance`, `uniformity`, `clean_cup`, `sweetness`, `overall` | Float | The ten sensory scores, 0-10. **Never features** |
| `total_cup_points` | Float | The target. Exactly the sum of the ten scores above |

## `clean.market_context` — one country and market year

USDA PSD pivoted wide. 4,616 rows. **Every value is in thousands of 60 kg bags**, and
null means "not reported", never zero.

| Column | Type | Meaning |
|---|---|---|
| `country` | String | PSD country name |
| `market_year` | Int | Marketing year as PSD labels it |
| `production` | Float? | Total green coffee produced |
| `arabica_production`, `robusta_production`, `other_production` | Float? | Production by species |
| `exports`, `imports` | Float? | Total trade |
| `bean_exports`, `bean_imports` | Float? | Green bean trade |
| `roast_ground_exports`, `roast_ground_imports` | Float? | Roasted and ground trade |
| `soluble_exports`, `soluble_imports` | Float? | Instant coffee trade |
| `domestic_consumption` | Float? | Consumed in country |
| `roast_ground_domestic_consumption`, `soluble_domestic_consumption` | Float? | Consumption by form |
| `beginning_stocks`, `ending_stocks` | Float? | Inventories |
| `total_supply`, `total_distribution` | Float? | PSD balance totals |

## `clean.boroughs` — one alcaldia of Mexico City

INEGI's 2020 geostatistical framework, layer `09mun`, reprojected to WGS84, with who
lives there from INEGI's 2020 Census (a total row per alcaldia; the 16 add up exactly to
the state's own total, which the build checks). 16 rows.

| Column | Type | Meaning |
|---|---|---|
| `borough_id` | String | INEGI's code (CVEGEO, entity + municipality); a question names a borough, so filter on `borough` |
| `borough` | String | Name as INEGI spells it |
| `area_km2` | Float | Area in the layer's own projection (conformal, so ~0.6% out: the 16 sum to 1,486 km² against the published 1,495) |
| `boundary` | Binary | The polygon as WKB in WGS84. Readable with `ST_GeomFromWKB`, or any GIS |
| `population` | Int? | People living there, 15 March 2020 (the census' day) |
| `adults` | Int? | People aged 18 and over |
| `households` | Int? | Inhabited private dwellings |
| `schooling_years` | Float? | Average years of schooling of people aged 15 and over |
| `economically_active` | Int? | People aged 12 and over working or looking for work |
| `workplaces` | Int? | DENUE's establishments of every activity in the borough (null without a DENUE token) |
| `jobs_estimate` | Float? | People working there, estimated from DENUE's staff-size bands (each band's middle; "251 or more" counted as 251, so a floor) - who is there by day, not who lives there |

Per inhabitant: divide by `population` (per 10,000: `* 10000.0 / population`). Per job:
divide by `jobs_estimate` (per 1,000: `* 1000.0 / jobs_estimate`).

## `clean.coffee_shops` — one place in DENUE's cafeterías class or OSM's cafés and ice-cream parlours

**Not every row is a coffee shop.** DENUE's class 722515 also holds juice stands, soda
fountains, ice-cream parlours and school tuck shops: for coffee shops, filter
`kind = 'coffee'`; count every row only when a question asks for every kind of place.

DENUE and OpenStreetMap side by side, each row placed in a borough by a point-in-polygon
join, given a `kind`, and linked to its twin in the other register when both list it.
11,217 rows (9,860 from DENUE; 1,357 from OSM, of which 232 are ice-cream parlours). The
registers are **not** merged: `matched_shop_id` says where they agree, so a count across
both can avoid counting one place twice.

| Column | Type | Meaning |
|---|---|---|
| `shop_id` | String | `denue-<id>` or `osm-<type>-<id>`, unique |
| `source` | String | `denue` (official register) or `osm` (crowd-sourced) |
| `name` | String? | As the source records it. DENUE shouts in capitals |
| `brand` | String? | OSM only: set when the place belongs to a chain |
| `employees_band` | String? | DENUE only: size band of the workforce, e.g. `0 a 5 personas` |
| `latitude`, `longitude` | Float | WGS84 |
| `borough_id`, `borough` | String? | From the spatial join. Null if the point falls outside every borough. Filter a borough by its name, `borough` |
| `declared_borough_id` | String? | The borough the source itself claims (DENUE's `AreaGeo`); null for OSM. The column the join is audited against |
| `kind` | String | What the place is: `coffee`, `tea`, `ice_cream`, `juice`, `soda_fountain`, `school`, `unnamed` or `unclassified`. For a coffee-only view, filter `kind = 'coffee'` |
| `kind_basis` | String | How the kind was decided: `name` (DENUE, read by the ordered rules in `cleaning.shop_kinds`) or `tag` (OSM, its own `amenity` tag) |
| `matched_shop_id` | String? | The same place in the other register, both ways: within 60 m, names at least 0.88 alike (Jaro-Winkler), each the other's best match |
| `listed_since` | Date? | DENUE only: the first day of the month of the register's edition the place entered it (`2024-11-01`). An entry, not an opening: 92% entered in the first edition (2010-07) or right after an economic census (2014-12, 2019-11, 2024-11) |

> DENUE's activity class 722515 is **wider than coffee**: 38% of it is named as coffee,
> 24% as juice stands, and ice-cream parlours are only 2%. Every row is kept with its
> `kind` rather than filtered. Where both registers list a place, the name rule's
> `coffee` agrees with OSM's tag 142 times out of 142, and finds 79% of the cafes OSM
> knows (`analysis.kind_scores`); `unclassified` names are where the rest hide.

## `clean.coffee_shop_history` — one place in one read of its register

Every read of DENUE and OpenStreetMap is kept (stage 4): `clean.coffee_shops` is each
register now, this table is each place at every read, one row per place per read day. A
place a later read no longer lists left the register; one that appears was added.
**For the places today, use `clean.coffee_shops`**; this table counts a place once per
read. `analysis.shop_turnover` counts what appeared and disappeared between reads.

| Column | Type | Meaning |
|---|---|---|
| `shop_id`, `source`, `name`, `kind`, `latitude`, `longitude`, `borough_id`, `borough` | | As in `clean.coffee_shops`, as that read listed them |
| `snapshot` | String | The read's day, `YYYY-MM-DD`. A read identical to the one before is not stored, so a snapshot is a day the register changed |

## `clean.mexico_production` — coffee grown in one Mexican municipality, one year

SIAP's closing agricultural statistics (definitive, not the monthly advance), crop
5710000 *Café cereza*, **every year from 2003 to 2025** - about 480 municipalities a year.
A question about one harvest filters `year` (the latest is 2025); a total over every
year is rarely what is asked. SIAP splits a municipality by district, CADER and water
regime; those rows are summed, and yield and price derived from the totals. Prices are
pesos of each year, not adjusted for inflation.

| Column | Type | Meaning |
|---|---|---|
| `year` | Int | Agricultural year of the closing statistics |
| `state_id`, `state` | String | INEGI state code (zero-padded) and name |
| `municipality_id` | String | INEGI CVEGEO (state + municipality), the key the boundary layers use |
| `municipality` | String | As SIAP spells it |
| `planted_ha`, `harvested_ha`, `lost_ha` | Float | Hectares planted, harvested, and lost to weather or pests |
| `production_t` | Float | Tonnes of **cherry**, not green coffee - several times the weight |
| `value_mxn` | Float | Value at the rural price, pesos |
| `yield_t_per_ha` | Float? | `production_t / harvested_ha`; null when nothing was harvested |
| `rural_price_mxn_per_t` | Float? | `value_mxn / production_t`; null when nothing was produced |

A state's rural price is its total `value_mxn` over its total `production_t`, and its
yield its total production over its total harvested area - never the average or the
maximum of its municipalities' own.

## `clean.roaster_coffees` — one coffee a roaster sells

What four Mexico City roasters list as coffee in their own shops (stage 3), read
2026-09-21: 167 coffees. Unlike the CQI, current and Mexican - but a shop's catalogue,
not a graded sample: nothing here was cupped by a third party.

| Column | Type | Meaning |
|---|---|---|
| `coffee_id` | String | `<shop>-<product id>`: the one-column key that joins the three roaster tables |
| `shop`, `product_id` | String | The shop, lower case as the config names it (`almanegra`, `buna`, `cucurucho`, `jiribilla` for Café con Jiribilla), and the platform's own product id |
| `title` | String | As the shop titles it |
| `url` | String | The product page |
| `description` | String? | The shop's own text, markup stripped; the source of RAG documents later |
| `origins` | Int | How many origins its sheet describes: 0 = no sheet, 2 or more = a blend |

## `clean.roaster_origins` — one origin a coffee's sheet describes

Read from "Label: value" lines or, on Buna, from headed paragraphs on the product page.
A blend lists each component in turn and gets a row for each. Canonical columns use the
vocabulary of a table that already exists, so they join: countries as PSD names them,
states as SIAP does, processing methods and varieties as the CQI spells them.

| Column | Type | Meaning |
|---|---|---|
| `coffee_id`, `shop`, `product_id`, `origin` | | Keys; `origin` is 1, 2, ... in the sheet's order |
| `country` | String? | PSD's name, from the country label, a state (which implies Mexico), or a place |
| `state` | String? | The Mexican state as SIAP names it (Estado de México is `México`) |
| `region` | String? | As written; the "origin" label when there is no region |
| `producer`, `farm` | String? | As written |
| `altitude_min_m`, `altitude_max_m` | Float? | The range written, in metres; equal for a single figure; numbers outside 300-3,000 m dropped |
| `varieties` | List[String]? | Lower-case, split on commas, "y", shares and slashes, spelled as the CQI spells them |
| `process` | String? | As the shop wrote it: "Natural Maceración Carbónica" |
| `processing_method` | String? | `washed`, `natural`, `honey`, `semi_washed`; `other` for experimental fermentations and lots sold as two methods; `unclassified` when no rule recognises the label |
| `species` | String? | `arabica` or `robusta`, only where the sheet says so |
| `sca_score` | Float? | The cupping score, where the shop publishes one |

## `clean.roaster_offers` — one coffee in one size

| Column | Type | Meaning |
|---|---|---|
| `offer_id` | String | `<shop>-<variant id>`: a platform's ids are only unique within a shop |
| `coffee_id`, `shop`, `product_id`, `variant_id` | String | Its coffee, and the platform's own ids |
| `variant_title` | String? | Size, grind or lot, as the shop titles the variant |
| `price_mxn` | Float | The listed price, pesos (Buna's pages state MXN; Squarespace's JSON does too) |
| `bag_grams` | Float? | From the titles: the variant's size, else the product's, times the bags in a pack. Never the platform's own weight, which contradicts the titles in 77 offers |
| `price_mxn_per_kg` | Float? | Null without a size, and for kits and samplers, whose price pays for more than coffee |
| `price_outlier` | Bool? | More than 3x off its product's median per kilogram: the three found were a price copied from another size. Kept as listed, flagged |
| `observed_on` | Date | When the catalogue was read: the first ingestion of this exact content, from the raw manifest (an unchanged catalogue read again keeps its date) |
| `snapshot` | String | That read, as the period the offer model's studies compare; stage 4's re-reads add more |

## `clean.roaster_flavors` — one tasting note a coffee's description names

What the shops say their coffees taste of, read from their own words ("Un café con notas
a chocolate y frambuesa") into the flavour categories of the SCA's descriptive
assessment (SCA-103). Read only after a cue ("notas a", "sabe a", "aroma", "en taza") and
before the end of its sentence, never from a producer's story; Cucurucho's titles are
read after the hyphen ("Chiapas- Caramelo, avellana y chocolate"). 94 of the 167 coffees
name at least one note; a coffee with none has no rows. The shops' claims, not a cupper's.

| Column | Type | Meaning |
|---|---|---|
| `coffee_id`, `shop` | String | The coffee, as in `roaster_coffees` |
| `note` | String | The note as the shop wrote it, in Spanish, lower case: `frambuesa`, `piloncillo`, `té negro` |
| `note_en` | String | The note in English: `raspberry`, `piloncillo`, `black tea`. One row per coffee and `note_en`: "jamaica" and "hibisco" are both `hibiscus` |
| `category` | String | `floral`, `fruity`, `sour_fermented`, `green_vegetative`, `other`, `roasted`, `nutty_cocoa`, `spice`, `sweet` (the form's Floral, Fruity, Sour/Fermented, Green/Vegetative, Other, Roasted, Nutty/Cocoa, Spice, Sweet) |
| `subcategory` | String? | Where the form has one: `berry`, `dried_fruit`, `citrus_fruit` (fruity); `musty_earthy`, `woody` (other); `cereal`, `burnt`, `tobacco` (roasted); `nutty`, `cocoa` (nutty_cocoa); `vanilla`, `brown_sugar` (sweet) |
| `source` | String | Where the shop wrote it: `title` or `description` |

Coffees that taste of a category: `COUNT(DISTINCT coffee_id)` filtered on `category`; its
share is over the coffees with any note, not over all 167. The share of the coffees with
notes that name one category (here `floral`):

```sql
SELECT count(DISTINCT CASE WHEN category = 'floral' THEN coffee_id END)::DOUBLE
       / count(DISTINCT coffee_id) AS share
FROM clean.roaster_flavors
```

For the origin or the process, join `roaster_origins` on `coffee_id` - only single-origin
coffees (one row there) have one - and `roaster_offers` for the price. The single-origin
coffees, to join and filter on only when the question names a country or a process:

```sql
SELECT coffee_id, any_value(country) AS country, any_value(processing_method) AS method
FROM clean.roaster_origins GROUP BY coffee_id HAVING count(*) = 1
```

## `clean.price_indicators` — one indicator, one day or month

The international price of green coffee (stage 4). The ICO's daily indicator prices
(its page shows the current month only, so every download is kept and stacked) and the
World Bank's monthly averages of two of them since 1960, converted from $/kg.

| Column | Type | Meaning |
|---|---|---|
| `period` | Date | The day, or the first day of the month a monthly average is for |
| `frequency` | String | `daily` (the ICO's) or `monthly` (the World Bank's) |
| `indicator` | String | `i_cip` (the ICO composite), `colombian_milds`, `other_milds`, `brazilian_naturals`, `robustas`. The World Bank publishes `other_milds` and `robustas` only |
| `usd_cents_per_lb` | Float | US cents per pound, ex-dock. Monthly values were $/kg with two decimals: ±0.23 cents/lb |
| `source` | String | `ico` or `world_bank` |
| `read_at` | Datetime (UTC) | The download the value came from; a day read twice keeps its latest reading |

A month's average is not the mean of its daily rows until the month is over: the ICO's
rows cover the days published so far.

## `clean.exchange_rates` — pesos per US dollar, one business day (Federal Reserve)

The Federal Reserve Bank of New York's noon buying rate (release H.10), through FRED,
daily since 8 November 1993. A day no rate was set (a US holiday) has no row.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The business day |
| `mxn_per_usd` | Float | Pesos per US dollar |

Green coffee in pesos per kilogram is `usd_cents_per_lb / 45.359237 * mxn_per_usd`; a
month's price goes with the mean of that month's rates (how FRED averages its own monthly
series):

```sql
SELECT p.period, p.indicator,
       p.usd_cents_per_lb / 45.359237 * avg(r.mxn_per_usd) AS mxn_per_kg
FROM clean.price_indicators p
JOIN clean.exchange_rates r ON date_trunc('month', r.date) = p.period
WHERE p.frequency = 'monthly'
GROUP BY p.period, p.indicator, p.usd_cents_per_lb
```

## `clean.roaster_offer_history` — one offer as one read of the catalogues found it

Every read of the shops is kept (stage 4): the tables above are the catalogue as it is
now, this one is what it was at each read - the same columns as `clean.roaster_offers`,
one row per offer per read. A day read twice is its later read. **For today's
catalogue, use `clean.roaster_offers`**; this table counts each offer once per read.

| Column | Type | Meaning |
|---|---|---|
| `observation_id` | String | `<offer_id>@<snapshot>`, unique: one offer in one read |
| `offer_id`, `coffee_id`, `shop`, `product_id`, `variant_id`, `variant_title` | String | As in `clean.roaster_offers`; `offer_id` repeats across reads |
| `price_mxn`, `bag_grams`, `price_mxn_per_kg`, `price_outlier` | | As that read listed them |
| `observed_on`, `snapshot` | | The read: its date, and the same as text |

## `clean.roaster_origin_history` — one origin as one read described it

`clean.roaster_origins` for every read, with its `snapshot`: a coffee's sheet can
change between reads, and a price is explained by the sheet of its own read.

## `clean.consumer_prices` — one shelf price of mass-market packaged coffee (PROFECO)

PROFECO's *Quién es Quién en los Precios* (stage 4): its staff price packaged coffee in
supermarkets, convenience stores, markets and pharmacies across Mexico, fortnight by
fortnight. Instant and roasted-and-ground coffee from national brands (Nescafé, Legal,
Internacional, Los Portales, store brands), not specialty coffee: that is
`clean.roaster_offers`. **Every fortnight since January 2024** (2024 and 2025 whole, the
current year to the month before last): a question about today's price filters `date`
(the last twelve months, say); a median over every row sets 2024 beside 2026.

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The day the price was recorded |
| `fortnight` | Date | The fortnight PROFECO files it under: the 1st or the 16th of its month |
| `product` | String | `instant` or `ground` (roasted and ground) |
| `brand` | String | As PROFECO writes it (`Nescafé. Clásico`, `Legal`) |
| `presentation` | String | As PROFECO writes it (`Frasco 170 Gr. Mezclado con Caramelo`) |
| `grams` | Float | The size the presentation states |
| `sweetened` | Bool | A blend with sugar or caramel ("Mezclado con ..."): its price per kilogram is per kilogram of both |
| `decaf` | Bool | Decaffeinated |
| `price_mxn` | Float | Pesos, for the jar, bag or sachet |
| `price_mxn_per_kg` | Float | Pesos per kilogram of what the presentation holds |
| `chain` | String | `Wal-mart`, `Hipermercado Soriana`, `Oxxo`, `Mercado Publico`... |
| `store_type` | String | `Supermercado / Tienda de Autoservicio`, `Tienda de Conveniencia`, `Farmacias`, `Mercados`, `Central de Abasto` |
| `store` | String | The store, with its branch (`Walmart Sucursal Polanco`) |
| `state` | String | As PROFECO spells it (`Ciudad de México`, `Estado de México`) |
| `municipality` | String | As the store declares it |
| `latitude`, `longitude` | Float | As PROFECO geocoded the store; 6 of the city's 120 stores fall in another borough than they declare |
| `borough_id`, `borough` | String? | The city's rows only: the borough the store declares, with `clean.boroughs`' key and spelling. Filter a borough by its name, `borough` |

Read with care: **a median, not a mean** (a promotion is one shelf); **per kilogram of
product** - instant is concentrated, so a kilogram of it makes several times the cups a
kilogram of ground coffee does; and a store can list two prices for one product on one
day (24 times in 2026), both kept. For the city's shelves, filter `state = 'Ciudad de
México'` or `borough_id IS NOT NULL`; across the country, filter no state. **Plain** coffee -
ground or instant with nothing else in it - is `NOT sweetened AND NOT decaf`. PROFECO
reports by fortnight: group by `fortnight` for its periods.

## `features.review_features` — model input

`clean.coffee_reviews` joined to the market context of **the previous market year**
(a lot graded in year Y sees Y-1, the latest one complete when it was cupped). The ten
sensory scores are dropped here, so no downstream consumer can pick them up.

| Column | Type | Meaning |
|---|---|---|
| `review_id`, `snapshot`, `grading_date` | | Keys, carried for joins and splitting |
| `country`, `variety`, `processing_method`, `color` | String? | Categorical features |
| `altitude_m`, `moisture_pct`, `category_one_defects`, `category_two_defects`, `quakers` | Float? | Numeric features |
| `ctx_production` | Float? | Origin country's production, previous market year |
| `ctx_arabica_share` | Float? | Arabica ÷ total production (null when production is 0) |
| `ctx_export_share` | Float? | Exports ÷ production; can exceed 1 with re-exports |
| `ctx_domestic_consumption` | Float? | Origin country's own consumption |
| `total_cup_points` | Float | Target |

## `predictions.review_predictions` — batch scores

| Column | Type | Meaning |
|---|---|---|
| `review_id`, `snapshot`, `grading_date` | | Keys back to `clean.coffee_reviews`: join on `review_id` for the lot's country, variety or actual score |
| `prediction` | Float | Predicted `total_cup_points` |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.offer_features` — price model input

`clean.roaster_offer_history` with a usable price (a size to divide by, no
`price_outlier`), each joined to what its coffee's sheet said in the same read: one row
per offer per read. A blend's origins are summarised per
attribute: the value they agree on, `multiple` where they differ, null where no origin
states it. The listed `price_mxn` is dropped: the target is it divided by the size.

| Column | Type | Meaning |
|---|---|---|
| `observation_id` | String | One offer in one read: `<offer_id>@<snapshot>` |
| `offer_id`, `snapshot`, `observed_on` | | The offer across reads, and the read |
| `coffee_id` | String | The group of the split: every size and every read of one coffee is on one side |
| `shop`, `country`, `state`, `processing_method`, `variety`, `producer` | String? | Categorical features; the origin ones summarised as above |
| `altitude_m` | Float? | Mean of the midpoints of the coffee's origin altitude ranges |
| `bag_grams` | Float? | The size, for the discount a bigger bag gets |
| `origins_n`, `varieties_n` | Float? | How many origins and varieties the sheet names; more than one origin is a blend |
| `variety_bourbon`, `variety_caturra`, `variety_colombia`, `variety_garnica`, `variety_gesha`, `variety_heirloom`, `variety_jember`, `variety_marsellesa`, `variety_mundo_novo`, `variety_oro_azteca`, `variety_pluma_mejorado`, `variety_ruiru_11`, `variety_sarchimor`, `variety_sl28`, `variety_sl34`, `variety_typica` | Float? | 1 when the coffee's sheet lists that variety, 0 when it lists others, **null when it lists none** - "not stated" is not "not a Gesha". One column per variety that appears in at least four coffees, chosen by frequency and never by price |
| `price_mxn_per_kg` | Float | Target |

## `features.green_price_features` — price forecast input

One indicator in one month, from `clean.price_indicators`' monthly rows (the World
Bank's): everything it may know is the history up to the month before it, looked up by
date. A month whose month before is missing is left out.

| Column | Type | Meaning |
|---|---|---|
| `month_id` | String | `<indicator>-<YYYY-MM>`, unique |
| `month` | Date | The first day of the month |
| `decade` | String | `1990s`, `2020s`: the period the studies compare |
| `indicator`, `calendar_month` | String | `other_milds` or `robustas`; `01` to `12`, for the harvest seasons |
| `price_last` | Float | US cents/lb, the month before |
| `change_last`, `change_2_back`, `change_3_back` | Float? | The last three month-on-month changes, % |
| `change_12m` | Float? | Over the twelve months to the month before, % |
| `gap_to_mean_12m` | Float? | The last price against its twelve-month mean, %; null without twelve months |
| `volatility_6m` | Float? | Standard deviation of the last six changes; null without six |
| `arabica_robusta_ratio` | Float? | Other mild Arabicas over Robustas, the month before |
| `change_pct` | Float | Target: the month's change from the month before, % |

## `predictions.green_price_predictions` — batch price change forecasts

| Column | Type | Meaning |
|---|---|---|
| `month_id`, `decade`, `month` | | Keys back to the feature table |
| `prediction` | Float | Predicted `change_pct` |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `predictions.offer_predictions` — batch price estimates

| Column | Type | Meaning |
|---|---|---|
| `observation_id`, `offer_id`, `snapshot`, `observed_on`, `coffee_id` | | Keys: one row per offer per read. Join `clean.roaster_offer_history` on `observation_id` for the shop, size or price listed in that read |
| `prediction` | Float | Predicted `price_mxn_per_kg` |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## Raw layer

`raw/<source>/ingested_at=<timestamp>/` holds each download **exactly as served**, next
to a manifest with the URL, sha256, size and ingestion time. Nothing is parsed there.
Sources: see the README. Two are kept whole rather than latest: the ICO's page and the
roasters' catalogues, whose every read is history. PROFECO's archive is the year so far,
so its latest download holds the year.

### Documents

Each document of the corpus is a source of its own, stored as the PDF or the article XML
it was served as. Read into a frame of parts, which is what its contract checks:

| Column | Type | Meaning |
|---|---|---|
| `document_id` | String | The document, as `documents:` names it in the config |
| `part` | Int | Page of a PDF or section of an article, in reading order; a page with no text is left out and the others keep their numbers, so a citation still points at the right page |
| `part_title` | String? | The section's heading, for an article; a PDF has none a parser can trust |
| `text` | String | The part's text, Unicode-normalised so a ligature ("coﬀee") is searchable |

The rest of what is known about a document - title, publisher, year, licence, language
and its topics - is config, not data, and travels with every chunk of it.
