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

## `clean.borough_profile` — one indicator of one alcaldia, 2025 (INEGI's Intercensal Survey)

Who lives in each borough now: INEGI's 2025 Intercensal Survey (October-November 2025, a
sample of 7.3 million dwellings, representative of every municipality). An estimate, not a
count: each comes with INEGI's 90% interval. 16 boroughs x 26 indicators.

| Column | Type | Meaning |
|---|---|---|
| `borough_id` | String | INEGI's code, as in `clean.boroughs` |
| `borough` | String | Name as INEGI spells it |
| `year` | Int | 2025 |
| `indicator` | String | One of the names below |
| `unit` | String | What one unit of `value` is |
| `value` | Float? | The estimate; null where the sample was too small |
| `ci_low` | Float? | Lower limit of the 90% interval |
| `ci_high` | Float? | Upper limit of the 90% interval |
| `cv` | Float? | Coefficient of variation, percent: under 15 precise, 15-30 acceptable, over 30 use with care |

Indicators (`percent` = per 100 of the group named):
- People: `population` (in private dwellings), `median_age` (years), `aging_index` (people
  60+ per 100 under 15), `dependency_ratio` (people under 15 or 65+ per 100 aged 15-64),
  `lived_in_another_state_2020_pct` (of people 5+).
- Schooling: `schooling_years` (people 15+), `higher_education_pct` (of people 15+).
- Work: `economically_active_pct` (of people 12+), `unemployment_pct` (of the economically
  active), `salaried_pct` and `self_employed_pct` (of the employed),
  `no_health_coverage_pct` (of everyone), `works_in_another_state_pct`,
  `commutes_by_metro_or_metrobus_pct`, `commute_1_to_2_hours_pct`,
  `commute_over_2_hours_pct` (of the employed; the Metro one of those who commute).
- Homes: `households`, `female_headed_households`, `dwellings` (private, lived in),
  `occupants_per_dwelling`, `occupants_per_room`, `owned_pct`, `rented_pct`,
  `internet_pct`, `car_pct`, `computer_pct` (of dwellings).

One borough's figure: `WHERE indicator = 'median_age' AND borough = 'Coyoacán'`. Every
borough on one indicator: `WHERE indicator = 'rented_pct' ORDER BY value DESC`. Never
average `value` across indicators; a share of the city needs the counts, not the shares.

## `clean.census_zones` — one urban AGEB of Mexico City (2020 Census)

An AGEB is a few dozen blocks: 2,431 of them, finer than the 16 boroughs. Shops and
stations carry their `zone_id`.

| Column | Type | Meaning |
|---|---|---|
| `zone_id` | String | INEGI's 13-character key: state, borough, locality, AGEB |
| `borough_id`, `borough` | String | Its borough |
| `ageb` | String | Its number as INEGI writes it, `048-2`: unique within its borough, not across them |
| `label` | String | How a person reads it: `Miguel Hidalgo · AGEB 048-2`. An AGEB has no name |
| `area_km2` | Float | Area |
| `boundary` | Binary | Polygon as WKB, WGS84 |
| `population` | Int | People, 2020 |
| `dwellings` | Int? | Private dwellings lived in |
| `schooling_years` | Float? | Average years of schooling, 15+; null where nobody lives |
| `economically_active`, `people_65_plus` | Int? | People |
| `dwellings_with_internet`, `dwellings_with_car`, `dwellings_with_computer` | Int? | Dwellings |

Null = withheld by INEGI. Coffee shops per zone: `JOIN clean.coffee_shops USING (zone_id)`.

## `clean.transit_stations` — one Metro or Metrobús station on one line

From the city's GTFS feed. 195 Metro (11 outside the city: no borough), 287 Metrobús.

| Column | Type | Meaning |
|---|---|---|
| `station_id` | String | `<system>-<line>-<name>` |
| `system` | String | `metro` or `metrobus` |
| `line` | String | `1`..`12`, `A`, `B` (Metro); `1`..`7` (Metrobús) |
| `station` | String | Name as the feed spells it |
| `latitude`, `longitude` | Float | WGS84 |
| `borough_id`, `borough` | String? | Where it is; null outside the city |
| `zone_id` | String? | Its urban AGEB (`clean.census_zones`) |

## `clean.transit_ridership` — entries on one day: a Metro station, or a whole Metrobús line

Metro since 2010 per station; Metrobús since 2005 per line only (`station_id` null).

| Column | Type | Meaning |
|---|---|---|
| `date` | Date | The day |
| `system` | String | `metro` or `metrobus` |
| `line` | String | As in `clean.transit_stations` |
| `station_id`, `station` | String? | The Metro station; null for Metrobús |
| `entries` | Int | People who entered; 0 = the station was closed |

A system's average day is the mean of its day totals: `SELECT avg(total) FROM (SELECT date,
sum(entries) AS total FROM clean.transit_ridership WHERE system = 'metro' GROUP BY date)`. One
station's average day: `avg(entries) FILTER (WHERE entries > 0)`.

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
| `zone_id` | String? | The urban AGEB it falls in (`clean.census_zones`); null outside every one |
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

## `clean.producer_prices` — what one country's farmers were paid for coffee, one year (FAOSTAT)

FAO's producer prices of green coffee, the year's value, 1991-2025; about 20 countries
a year lately, not every grower. Countries use PSD's names (`Vietnam`, `Cote d'Ivoire`).

| Column | Type | Meaning |
|---|---|---|
| `country`, `year` | String, Int | The country and calendar year |
| `usd_per_t` | Float? | US dollars per tonne |
| `lcu_per_t` | Float? | Local currency per tonne |
| `price_index` | Float? | FAO's index, 2014-2016 = 100 |
| `flag` | String? | `A` official, `E` estimated, `X` from another organisation |
| `cherry` | Bool | The tonne is of **cherry**, not green coffee (Mexico: its figure is SIAP's rural price) |

A farmer's share of the port price is in `analysis.farmgate_prices`
(`share_of_benchmark`: the price over the World Bank's yearly price for the country's
mix of arabica and robusta); countries with `cherry` are not in it.

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

## `clean.household_coffee` — one household of INEGI's 2024 income and expenditure survey (ENIGH)

91,414 households in the sample, standing for 38.8 million; representative of each state,
not of a borough. What a household spent on coffee **to drink at home**, bought in the
survey's week and stated per quarter (coffee in a coffee shop is not here).

| Column | Type | Meaning |
|---|---|---|
| `year` | Int | The survey's year, 2024 |
| `household_id` | String | The survey's dwelling and household |
| `state_id`, `state` | String | INEGI's code and the state's name (`Ciudad de México`; `México` is the State of Mexico) |
| `municipality_id` | String | State + municipality, 5 digits |
| `stratum`, `psu` | String | The survey's design: stratum and primary sampling unit |
| `weight` | Int | How many households this one stands for |
| `members` | Int | People in it |
| `income_quarter_mxn` | Float | Current income in the quarter, pesos |
| `income_decile` | Int | 1 (the country's poorest tenth of households) to 10 |
| `instant_quarter_mxn`, `ground_quarter_mxn`, `prepared_quarter_mxn` | Float | Pesos paid in the quarter for instant coffee, beans or ground coffee, preparations for coffee drinks; 0 if none |
| `own_harvest_quarter_mxn` | Float | Coffee taken from its own harvest, valued in pesos |

Every figure is weighted: a share of households is `sum(weight * (condition)::INT) /
sum(weight)`; spending per household a month is `sum(weight * spent) / sum(weight) / 3`.
Never count rows.

## `clean.consumer_price_index` — Mexico's consumer price index, one month (INEGI's INPC)

| Column | Type | Meaning |
|---|---|---|
| `month` | Date | The month's first day, January 1969 on |
| `index` | Float | The general index; the second half of July 2018 = 100 |

A price of month A in pesos of month B: `price * index_B / index_A`. Empty when the INPC
token is not set.

## `clean.cup_of_excellence` — one lot of a year's Cup of Excellence Mexico, and its auction

Mexico's best lots each year (2012-2026; none in 2016 or 2020), judged blind by a national
and an international jury, then sold at an online auction. Green coffee, US dollars.

| Column | Type | Meaning |
|---|---|---|
| `year` | Int | The competition's year |
| `lot_id` | String | `<year>-<n>`, page order |
| `rank` | String? | As published: `1`, `1a`, `nw` (national winner); null for a national winner listed without one |
| `national_winner` | Bool | Scored below the international round (86-87 points), sold in its own auction |
| `score` | Float | Cup score, 0-100 |
| `farm`, `farmer`, `region` | String | As published |
| `state` | String? | The state, in SIAP's names; null where the region does not tell |
| `varieties` | List[String]? | Lower-case, as the roasters' sheets spell them (`gesha`); null before 2018 |
| `processing_method` | String? | `washed`, `natural`, `honey`, `semi_washed`, `other`; null before 2018 |
| `weight_kg` | Float? | The lot's green coffee, kg |
| `price_usd_per_lb` | Float? | The auction's high bid, US dollars a pound of green coffee; null if unsold |
| `total_usd` | Float? | What the lot fetched |
| `buyers` | String? | Who bought it |

A pound is 0.4536 kg: US$ per kg = `price_usd_per_lb * 2.20462`. Prices rose year by
year: compare within a year, or with the year's market (`clean.price_indicators`).

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
| `held_out_prediction` | Float | The same, from the champion's recipe refitted without the offer's coffee: what the price looks like to a model that never saw it. Residuals that rank offers use this one |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.zones_features` — coffee shops per urban AGEB, model input

One urban AGEB of `clean.census_zones` with what the census says of it, the stations and
the food places in it, and its borough's jobs per resident (`domains/coffee/zone_profile.py`,
the same table the zone studies read).

| Column | Type | Meaning |
|---|---|---|
| `zone_id` | String | The AGEB's 13-character key |
| `census`, `census_day` | | `2020` and the census' reference date, 2020-03-15 |
| `borough_id` | String | The group of the split: a borough is on one side only |
| `population` | Float | Residents |
| `people_per_km2` | Float? | Residents per km² of the AGEB |
| `schooling_years` | Float? | Mean years of schooling of people 15 and over; null where INEGI withheld it |
| `active_pct`, `aged_65_plus_pct` | Float? | Economically active people, people 65 and over, % of residents |
| `internet_pct`, `car_pct`, `computer_pct` | Float? | Lived-in dwellings with each, % |
| `metro_stations`, `metrobus_stations` | Float | Stations inside the AGEB |
| `other_places` | Float | DENUE's juice bars, ice cream parlours, soda fountains and tea houses inside it (by name; unclassified places are not counted) |
| `borough_jobs_per_resident` | Float? | The borough's estimated jobs (DENUE) over its residents: the daytime population |
| `coffee_shops` | Float | Target: DENUE's places in the AGEB whose name says coffee shop |

## `predictions.zones_predictions` — expected coffee shops per AGEB

| Column | Type | Meaning |
|---|---|---|
| `zone_id`, `census`, `census_day`, `borough_id` | | Keys back to `clean.census_zones` |
| `prediction` | Float | Expected `coffee_shops` for a zone like it |
| `held_out_prediction` | Float | The same, from a model fitted without the zone's borough: **the one to set against the zone's own count**. Expected minus listed is how many coffee shops a zone is missing |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.green_range_features` — green coffee price outlook input

One indicator, one month and one horizon (3, 6 or 12 months) from `clean.price_indicators`'
monthly rows: the change from the price `horizon_months` before, knowing only what was
published then. The months 3, 6 and 12 after the last published one are rows too, with
no price and no target: the outlook.

| Column | Type | Meaning |
|---|---|---|
| `outlook_id` | String | `<indicator>-<YYYY-MM>-<h>m`, unique |
| `month` | Date | The month priced (its first day) |
| `year` | String | Its year: the block the gate resamples whole |
| `indicator`, `calendar_month` | String | `other_milds` or `robustas`; `01` to `12` |
| `horizon_months` | Float | 3, 6 or 12: how long before the month the forecast is made |
| `price_last` | Float? | US cents/lb, the month the forecast is made in |
| `change_last`, `change_2_back`, `change_3_back`, `change_12m`, `gap_to_mean_12m`, `volatility_6m`, `arabica_robusta_ratio` | Float? | As in `features.green_price_features`, known in that month |
| `change_pct` | Float? | Target: the change over the horizon, %; null for the months ahead |

## `predictions.green_range_predictions` — green coffee price ranges

| Column | Type | Meaning |
|---|---|---|
| `outlook_id`, `year`, `month` | | Keys back to the feature table; join it for `price_last` and `horizon_months` |
| `prediction` | Float | Predicted `change_pct` |
| `lower`, `upper` | Float | The range meant to hold the change four times in five, % |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.shelf_price_features` — fair shelf price input

One product in one store on one day of `clean.consumer_prices` (two readings of it the
same day, at their mean price), its price in pesos of the latest month of
`clean.consumer_price_index`.

| Column | Type | Meaning |
|---|---|---|
| `reading_id` | String | `<store>|<product_name>|<YYYY-MM-DD>` |
| `date`, `month` | | The day read, and its month `YYYY-MM` |
| `product_name` | String | Brand, size and kind as one name: `nescafe clasico 200 g sweetened` |
| `brand`, `chain`, `store_type`, `state` | String? | As PROFECO writes them, lower case and without accents or punctuation |
| `product` | String | `instant` or `ground` |
| `grams` | Float | The size |
| `sweetened`, `decaf` | Float | 1 or 0 |
| `price_today_mxn` | Float | Target: the price in pesos of the index's latest month (nominal without the index) |

## `predictions.shelf_price_predictions` — fair shelf prices

| Column | Type | Meaning |
|---|---|---|
| `reading_id`, `month`, `date` | | Keys back to the feature table; join it for the price asked |
| `prediction` | Float | The fair `price_today_mxn`: under it is a deal, over it a markup. Readings from 2026 on were never learned from |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.shop_kind_features` — is a DENUE place a coffee shop, input

One DENUE place of `clean.coffee_shops`: the words of its name besides the ones the kind
rules read, its staff, when it was registered, and its zone.

| Column | Type | Meaning |
|---|---|---|
| `shop_id`, `listed_since`, `listed_year` | | Keys: the place, its registration date and year |
| `block` | String | The group of the split: its AGEB (`none-<shop_id>` outside one) |
| `employees_band`, `borough_id` | String? | Categorical features |
| `listed_since_year` | Float | Year it entered the register |
| `name_word_count` | Float | Words in its name |
| `name_puesto`, `name_venta`, `name_desayunos`, `name_crepas`, `name_snacks`, `name_tortas`, `name_restaurante`, `name_kreme`, `name_krispy`, `name_turno`, `name_postres`, `name_ensaladas`, `name_casa`, `name_barra`, `name_antojitos`, `name_pasteleria`, `name_cocteles`, `name_cocina`, `name_dulce`, `name_matutino`, `name_creperia`, `name_gourmet`, `name_pan`, `name_comida`, `name_tierra`, `name_plaza`, `name_neverias`, `name_refresquerias`, `name_similares`, `name_oasis` | Float | 1 when the name has the word: the thirty most frequent words no kind rule reads, chosen by count |
| `people_per_km2`, `schooling_years`, `internet_pct`, `aged_65_plus_pct`, `metro_stations`, `metrobus_stations` | Float? | Its AGEB's, as in `features.zones_features` |
| `is_coffee` | Float? | Target: 1 a coffee shop by name, 0 another kind; **null where no rule classified it** - those are scored |

## `predictions.shop_kind_predictions` — how likely each place is a coffee shop

| Column | Type | Meaning |
|---|---|---|
| `shop_id`, `listed_year`, `listed_since`, `block` | | Keys back to `clean.coffee_shops` |
| `prediction` | Float | Probability the place is a coffee shop. Summed over the unclassified places, how many coffee shops they hold |
| `held_out_prediction` | Float | The same from a model that never saw the place's zone; equal to `prediction` for unclassified places |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.auction_features` — Cup of Excellence auction input

One lot of `clean.cup_of_excellence`, with the median price of the other lots of its
auction and the year's mean price of other mild Arabicas.

| Column | Type | Meaning |
|---|---|---|
| `lot_id`, `auction`, `auction_year` | | Keys: the lot, its year as text and as January 1 |
| `state`, `processing_method` | String? | Categorical features |
| `score_band` | String | `under 87`, `87-88`, `88-89`, `89-90`, `90 and over`: the baseline's groups |
| `score` | Float | The jury's score |
| `national_winner`, `gesha` | Float | 1 or 0 |
| `weight_kg`, `varieties_n` | Float? | The lot's weight; how many varieties it lists |
| `market_usd_per_lb` | Float? | Other mild Arabicas, the year's mean monthly price, US dollars a pound |
| `auction_median_usd_per_lb` | Float | The median price of the other lots sold at its auction (the latest auction's for a year not held yet), US dollars a pound |
| `premium_pct` | Float? | Target: the lot's price over that median, %; null for an unsold lot |

## `predictions.auction_predictions` — expected auction premiums

| Column | Type | Meaning |
|---|---|---|
| `lot_id`, `auction`, `auction_year` | | Keys back to `clean.cup_of_excellence` |
| `prediction` | Float | Predicted `premium_pct`; the price is `auction_median_usd_per_lb` × (1 + it / 100) |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `features.households_features` — who buys coffee, input

One household of `clean.household_coffee`.

| Column | Type | Meaning |
|---|---|---|
| `household_id`, `survey`, `survey_year` | | Keys: the household, the survey's year as text and as January 1 |
| `psu` | String | The group of the split: the survey's primary sampling unit, neighbours on one side |
| `state` | String | Categorical feature |
| `members`, `income_quarter_mxn`, `income_per_member_mxn` | Float | Its size and income in the quarter, in total and per member |
| `grows_coffee` | Float | 1 when it took coffee from its own harvest |
| `buys_coffee` | Float | Target: 1 when it paid for any coffee in the week recorded |

## `predictions.households_predictions` — how likely each household buys coffee

| Column | Type | Meaning |
|---|---|---|
| `household_id`, `survey`, `survey_year`, `psu` | | Keys back to `clean.household_coffee`: join for its `weight` and `municipality_id` |
| `prediction` | Float | Probability it bought coffee in the week |
| `held_out_prediction` | Float | The same from a model that never saw its sampling unit: the one to average by area |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## `analysis.price_ladder` — a kilogram of coffee at one step from the farm to the shelf

Each step is its own product - cherry, green coffee, a supermarket's ground or instant, a
roaster's bag - so a gap between steps is not a margin. Six rows.

| Column | Type | Meaning |
|---|---|---|
| `step` | String | `cherry at the farm gate`, `green coffee at the port`, `supermarket, ground`, `supermarket, ground + sugar`, `supermarket, instant`, `specialty roaster` |
| `source` | String | Where the price comes from: SIAP, World Bank and FRED, PROFECO, the roasters |
| `unit` | String | What a kilogram is of: `kg of coffee cherry`, `kg of green coffee`, `kg of roasted coffee`... |
| `measure` | String | How the price was summarised: `median`, `the month's price`, `value over volume` |
| `mxn_per_kg` | Float | Pesos per kilogram |
| `observations` | Int | Prices behind it |
| `period` | String | When: a year (`2025`), a month (`2026-08`), a span or a day |

## `analysis.green_coffee_in_pesos` — an international green coffee price in pesos, one month

The World Bank's monthly prices at the month's mean exchange rate (FRED). Since 1993.

| Column | Type | Meaning |
|---|---|---|
| `period` | Date | First day of the month |
| `indicator` | String | `other_milds` (the group Mexico's washed Arabicas trade in) or `robustas` |
| `usd_cents_per_lb` | Float | US cents a pound |
| `mxn_per_usd` | Float | Pesos per dollar, the month's mean |
| `rate_days` | Int | Business days the exchange rate is the mean of |
| `mxn_per_kg` | Float | Pesos per kilogram of green coffee |

## `analysis.real_prices` — a step's price in pesos of one month, deflated by the INPC

| Column | Type | Meaning |
|---|---|---|
| `step` | String | `cherry at the farm gate` (yearly), `green coffee at the port`, `ground coffee on a shelf` (monthly) |
| `period` | Date | The year's (1 January) or the month's first day |
| `frequency` | String | `annual` or `monthly` |
| `nominal_mxn_per_kg` | Float | Pesos per kilogram of the day |
| `real_mxn_per_kg` | Float | The same in pesos of `pesos_of` |
| `pesos_of` | Date | The month whose pesos `real_mxn_per_kg` is in: the INPC's latest |

## `analysis.price_transmission` — how much of a move in green coffee reaches a step's price

The cumulative elasticity after each lag: 0.18 means a 1% rise in green coffee (in pesos)
has moved the step's price 0.18% by then. 95% interval from a block bootstrap.

| Column | Type | Meaning |
|---|---|---|
| `step` | String | `ground coffee on a shelf`, `instant coffee on a shelf` (monthly, PROFECO), `cherry at the farm gate` (yearly, SIAP) |
| `frequency` | String | `monthly` or `annual` |
| `lag` | Int | Months (or years) after the move: 0 is the same month |
| `pass_through` | Float | The elasticity, cumulative to that lag |
| `ci_low`, `ci_high` | Float | Its 95% interval: one that holds 0 says no pass-through is shown |
| `changes` | Int | Price changes it was estimated from |

## `analysis.consumer_prices_by_borough` — the median shelf price of one coffee line in one borough

PROFECO's prices over the last twelve months of its survey; 13 of the 16 boroughs have
shelves in it.

| Column | Type | Meaning |
|---|---|---|
| `borough_id`, `borough` | String | The borough |
| `line` | String | `ground`, `instant`, each also `, sweetened` or `, decaf` |
| `median_mxn_per_kg` | Float | Median pesos per kilogram |
| `prices`, `stores` | Int | Readings and stores behind it |

## `analysis.consumer_prices_by_fortnight` — the median shelf price of one coffee line, one fortnight

| Column | Type | Meaning |
|---|---|---|
| `scope` | String | `city` (Mexico City's shelves) or `national` |
| `line` | String | As in `analysis.consumer_prices_by_borough` |
| `fortnight` | Date | Its first day |
| `median_mxn_per_kg` | Float | Median pesos per kilogram |
| `prices`, `stores` | Int | Readings and stores behind it |

## `analysis.market_summary` — the world's top coffee producers in the latest market year

USDA PSD. **Volumes are in thousands of 60 kg bags.**

| Column | Type | Meaning |
|---|---|---|
| `country` | String | PSD's name |
| `production` | Float | Thousands of 60 kg bags |
| `world_share_pct` | Float | Its share of the world's production, % |
| `export_ratio` | Float | Exports over production, a fraction (0.6 is 60%) |
| `domestic_consumption` | Float | Thousands of 60 kg bags |
| `imported_share_of_use` | Float | Imports over domestic consumption, a fraction |

## `analysis.household_coffee_by_state` — who buys coffee to drink at home, one state (ENIGH 2024)

Weighted by the survey's expansion factors; the first row (`state` = `Mexico`, no
`state_id`) is the country. **Every `_share` is a fraction from 0 to 1**, not a percent.

| Column | Type | Meaning |
|---|---|---|
| `state_id`, `state` | String | INEGI's state; `Ciudad de México` is the city |
| `households_sampled`, `households` | Int | Households surveyed, and the households they stand for |
| `bought_share` | Float | Households that bought coffee in the survey's week, a fraction |
| `bought_share_low`, `bought_share_high` | Float | Its 90% interval |
| `instant_share`, `ground_share`, `prepared_share` | Float | Households that bought each kind, a fraction |
| `monthly_mxn` | Float | Pesos a month a household spends on coffee, every household counted |
| `monthly_mxn_low`, `monthly_mxn_high` | Float | Its 90% interval |
| `monthly_per_buyer_mxn` | Float | Pesos a month among the households that bought |
| `income_per_mille` | Float | Coffee spending per 1,000 pesos of income |
| `own_harvest_share` | Float | Households that drank coffee they grew, a fraction |

## `analysis.household_coffee_by_decile` — the same, by income decile

| Column | Type | Meaning |
|---|---|---|
| `area` | String | `Mexico` (the country) or `Ciudad de México` |
| `income_decile` | Int | 1 is the poorest tenth of households, 10 the richest |
| `households_sampled`, `households` | Int | Households surveyed, and the households they stand for |
| `bought_share` | Float | Households that bought coffee in the survey's week, a fraction from 0 to 1 |
| `bought_share_low`, `bought_share_high` | Float | Its 90% interval |
| `instant_share`, `ground_share`, `prepared_share` | Float | Households that bought each kind, a fraction |
| `monthly_mxn` | Float | Pesos a month a household spends on coffee, every household counted |
| `monthly_mxn_low`, `monthly_mxn_high` | Float | Its 90% interval |
| `monthly_per_buyer_mxn` | Float | Pesos a month among the households that bought |
| `income_per_mille` | Float | Coffee spending per 1,000 pesos of income |
| `own_harvest_share` | Float | Households that drank coffee they grew, a fraction |

## `analysis.coe_by_year` — one year of Mexico's Cup of Excellence

| Column | Type | Meaning |
|---|---|---|
| `year` | Int | The competition's year |
| `lots`, `sold` | Int | Lots ranked, and sold at auction |
| `median_score` | Float | The jury's median score |
| `median_usd_per_lb`, `top_usd_per_lb` | Float | The median and the highest price, US dollars a pound |
| `auction_usd` | Float | What the auction raised, US dollars |
| `commodity_usd_per_lb` | Float | The year's mean other mild Arabicas price, US dollars a pound |
| `mxn_per_usd` | Float | The year's mean exchange rate |
| `median_times_commodity` | Float | The median lot's price over the commodity's |
| `median_mxn_per_kg` | Float | The median lot's price in pesos a kilogram |

## `analysis.coe_score_price` — what a point of score is worth at the Cup of Excellence

One row: the percent a point of score adds to a lot's price, each year its own level.

| Column | Type | Meaning |
|---|---|---|
| `lots`, `years` | Int | Lots and years it was estimated from |
| `percent_per_point` | Float | Percent a point adds to the price |
| `percent_low`, `percent_high` | Float | Its 95% interval |
| `rank_correlation` | Float | Spearman's rank correlation of score and price within years |
| `rank_correlation_low`, `rank_correlation_high` | Float | Its 95% interval |

## `analysis.flavor_profiles` — how often the roasters' coffees name a flavour category

The share of single-origin coffees whose description names each SCA category.

| Column | Type | Meaning |
|---|---|---|
| `dimension` | String | What the coffees are grouped by: `all`, `origin`, `process`, `shop` |
| `group` | String | The group: `every coffee` (for `all`), `Mexico` or `elsewhere`, `washed`, `natural`, `other`, or a shop |
| `coffees` | Int | Coffees in the group |
| `category` | String | `floral`, `fruity`, `sweet`, `nutty_cocoa`, `spice`, `roasted`, `sour_fermented`, `green_vegetative`, `other` |
| `share_pct` | Float | Coffees naming it, % of the group |

## `analysis.borough_coffee_shops` — one borough's coffee shops against its area, people and jobs

DENUE's coffee shops (`kind = 'coffee'`).

| Column | Type | Meaning |
|---|---|---|
| `borough_id`, `borough` | String | The borough |
| `area_km2` | Float | Its area |
| `population` | Int | Residents, 2020 Census |
| `schooling_years` | Float | Mean years of schooling, 15 and over |
| `workplaces`, `jobs_estimate` | | DENUE's workplaces of every activity, and their staff estimated from its bands |
| `coffee_shops` | Int | DENUE's coffee shops |
| `per_km2`, `per_10k_people`, `per_1k_jobs` | Float | Coffee shops per km², per 10,000 residents, per 1,000 jobs |
| `jobs_per_resident` | Float | The daytime population against the night-time one |

## `analysis.zone_coffee_correlations` — which trait of an urban AGEB goes with its coffee shops

Spearman's rank correlation of each trait with the AGEB's coffee shops per km², over the
2,431 AGEBs, with a 95% interval.

| Column | Type | Meaning |
|---|---|---|
| `trait` | String | `schooling_years`, `people_per_km2`, `internet_pct`, `car_pct`, `computer_pct`, `aged_65_plus_pct` |
| `zones` | Int | AGEBs with the trait known |
| `rho` | Float | The correlation, -1 to 1 |
| `rho_low`, `rho_high` | Float | Its 95% interval |

## `analysis.chain_strategies` — what a supermarket chain charges for coffee, and how often it cuts

| Column | Type | Meaning |
|---|---|---|
| `chain`, `store_type` | String | PROFECO's chain and kind of store |
| `readings`, `stores` | Int | Prices and stores behind it |
| `price_index` | Float | Its price against every chain's for the same product the same month: 1.05 is 5% dearer |
| `ci_low`, `ci_high` | Float | Its 95% interval |
| `cut_pct` | Float | Readings under 90% of what the same store asked that quarter, % |
| `strategy` | String | `premium`, `discount` or `at the market`, and `promotional` or `steady` |

## `analysis.production_by_state` — coffee grown in one Mexican state, one year (SIAP)

| Column | Type | Meaning |
|---|---|---|
| `year` | Int | The harvest's year |
| `state` | String | SIAP's state |
| `municipalities` | Int | Municipalities that grew coffee |
| `planted_ha` | Float | Hectares planted |
| `production_t` | Float | Tonnes of coffee cherry |
| `value_mxn` | Float | Its value, pesos |
| `share_pct` | Float | The state's share of the year's national production, % |
| `rural_price_mxn_per_t` | Float | Pesos per tonne of cherry at the farm gate |

## `analysis.price_outlook` — where a green coffee price could be 3, 6 and 12 months ahead

The range its own changes over each horizon have spanned, four times in five, since the
series began, set on the last published price - the range the `green_range` model is held
against (and is served while no model beats it).

| Column | Type | Meaning |
|---|---|---|
| `indicator` | String | `other_milds` or `robustas` |
| `horizon_months` | Int | 3, 6 or 12 |
| `from_month`, `to_month` | Date | The last published month, and the one the horizon reaches |
| `price_now` | Float | The last published price, US cents a pound |
| `change_low_pct`, `change_median_pct`, `change_high_pct` | Float | The changes at the range's low end, middle and high end, % |
| `months` | Int | Historical changes the range was drawn from |
| `price_low`, `price_median`, `price_high` | Float | The same as prices, US cents a pound |

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
