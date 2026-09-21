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
join. 10,985 rows (9,860 from DENUE, 1,125 from OSM). The registers are **not**
deduplicated against each other: they disagree about what exists, and choosing between
them is analysis, not cleaning.

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

> DENUE's activity class 722515 is **wider than coffee**: it covers soda fountains and
> ice-cream parlours too. Every row is kept and labelled rather than filtered by name,
> because no filter has been measured yet.

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
