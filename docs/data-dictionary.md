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
| `country` | String | Origin, renamed to match USDA PSD (Hawaii and Puerto Rico → United States) |
| `region` | String? | Growing region, free text as submitted |
| `variety` | String? | Lower-cased; blends and "unknown" are kept as written |
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

INEGI's 2020 geostatistical framework, layer `09mun`, reprojected to WGS84. 16 rows.

| Column | Type | Meaning |
|---|---|---|
| `borough_id` | String | Official CVEGEO: entity + municipality, e.g. `09015` is Cuauhtémoc |
| `borough` | String | Name as INEGI spells it |
| `area_km2` | Float | Area in the layer's own projection (conformal, so ~0.6% out: the 16 sum to 1,486 km² against the published 1,495) |
| `boundary` | Binary | The polygon as WKB in WGS84. Readable with `ST_GeomFromWKB`, or any GIS |

## `clean.coffee_shops` — one place that sells coffee

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
| `borough_id`, `borough` | String? | From the spatial join. Null if the point falls outside every borough |
| `declared_borough_id` | String? | The borough the source itself claims (DENUE's `AreaGeo`); null for OSM. The column the join is audited against |
| `kind` | String | What the place is: `coffee`, `tea`, `ice_cream`, `juice`, `soda_fountain`, `school`, `unnamed` or `unclassified`. For a coffee-only view, filter `kind = 'coffee'` |
| `kind_basis` | String | How the kind was decided: `name` (DENUE, read by the ordered rules in `cleaning.shop_kinds`) or `tag` (OSM, its own `amenity` tag) |
| `matched_shop_id` | String? | The same place in the other register, both ways: within 60 m, names at least 0.88 alike (Jaro-Winkler), each the other's best match |

> DENUE's activity class 722515 is **wider than coffee**: 38% of it is named as coffee,
> 24% as juice stands, and ice-cream parlours are only 2%. Every row is kept with its
> `kind` rather than filtered. Where both registers list a place, the name rule's
> `coffee` agrees with OSM's tag 142 times out of 142, and finds 79% of the cafes OSM
> knows (`analysis.kind_scores`); `unclassified` names are where the rest hide.

## `clean.mexico_production` — coffee grown in one Mexican municipality, one year

SIAP's closing agricultural statistics (definitive, not the monthly advance), crop
5710000 *Café cereza*. 489 rows for 2025. SIAP splits a municipality by district, CADER
and water regime; those rows are summed, and yield and price derived from the totals.

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

## `clean.roaster_coffees` — one coffee a roaster sells

What four Mexico City roasters list as coffee in their own shops (stage 3), read
2026-09-21: 167 coffees. Unlike the CQI, current and Mexican - but a shop's catalogue,
not a graded sample: nothing here was cupped by a third party.

| Column | Type | Meaning |
|---|---|---|
| `shop`, `product_id` | String | The shop (as named in the config) and the platform's own product id: the key |
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
| `shop`, `product_id`, `origin` | | Keys; `origin` is 1, 2, ... in the sheet's order |
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
| `shop`, `product_id`, `variant_id` | String | Keys; `variant_id` is the platform's |
| `variant_title` | String? | Size, grind or lot, as the shop titles the variant |
| `price_mxn` | Float | The listed price, pesos (Buna's pages state MXN; Squarespace's JSON does too) |
| `bag_grams` | Float? | From the titles: the variant's size, else the product's, times the bags in a pack. Never the platform's own weight, which contradicts the titles in 77 offers |
| `price_mxn_per_kg` | Float? | Null without a size, and for kits and samplers, whose price pays for more than coffee |
| `price_outlier` | Bool? | More than 3x off its product's median per kilogram: the three found were a price copied from another size. Kept as listed, flagged |

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
| `review_id`, `snapshot`, `grading_date` | | Keys back to the feature table |
| `prediction` | Float | Predicted `total_cup_points` |
| `model_version` | String | Registry version that produced the row |
| `predicted_at` | Datetime (UTC) | When the batch job ran |

## Raw layer

`raw/<source>/ingested_at=<timestamp>/` holds each download **exactly as served**, next
to a manifest with the URL, sha256, size and ingestion time. Nothing is parsed there.
Sources: `cqi_2018`, `cqi_2023`, `psd_coffee` (see the README).
