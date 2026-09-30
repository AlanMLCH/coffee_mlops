# Data sources: what was verified about each

The reference for every source the coffee domain reads: where it lives, how it is
reached, what the file really looks like and the traps found in it. Each fact was
checked with a real request on the date given. An address, a code, a column or a
parameter not written here (or in the YAML's comments) was not verified, and is checked
before it is used. What each source *says* - the findings, the tables, the figures - is in
the [development plan](development_plan.md); the columns each clean table ends up with are
in the [data dictionary](../src/domains/coffee/data_dictionary.md).

## Rules every source follows

- **Raw is the source as served.** Each download lands in `raw/<source>/` untouched, with
  its sha256, size and ingestion time in a manifest, in a partition of its own only when
  its bytes changed. Every check of a source, changed or not, is a line in
  `raw/<source>/checks.jsonl`. Raw is never pruned.
- **Canonical before stored.** A source that answers the same rows in another order
  (DENUE) or with another layout (Buna's pages) is sorted and serialised canonically
  first, or every run would look like new data.
- **Polite, and refusals respected.** robots.txt is read before any page, a
  `Crawl-delay` is honoured, the user agent names the project. A 403, a CAPTCHA or a
  robots.txt that forbids the path is not read around: the source is skipped out loud, and
  a document behind one is fetched by hand into `data/<domain>/inbox/documents/`.
- **Credentials never reach a URL that is logged, a cache key or a manifest.** Every
  script that touches an API goes through the project's client (`ApiClient`,
  `http_client`), which silences the request logger - DENUE carries its token in the URL
  path.
- **Freshness is declared.** `refresh_hours` per source (a file is not downloaded again
  before it), `cache_hours` per API (required: the default would be "forever").
- **Google Places is never persisted.** Its terms restrict storing and caching; it is not
  a source here.

## Files

### CQI cupping scores: `cqi_2018`, `cqi_2023`

Two public scrapes of the Coffee Quality Institute's database, unioned (verified
2026-09-18).

| Snapshot | Address | Rows | Graded | Access |
|---|---|---|---|---|
| 2018 | `jldbc/coffee-quality-database`, `data/arabica_data_cleaned.csv`, pinned to a commit | 1,311 arabicas, 44 columns | 2010-2018 | GitHub raw, MIT |
| 2023 | `https://www.kaggle.com/api/v1/datasets/download/fatihb/coffee-quality-data-cqi` → `df_arabica_clean.csv` | 207 arabicas, 41 columns | 2022-2023 (harvests 2021-2023) | ZIP, no token (302 to a signed URL, never stored) |

- **Frozen in May 2023.** No newer public version exists; the original database needs a
  login. A bootstrap, never the source of recency - said wherever it is used.
- Different schemas for the same item (`Harvest.Year` / `Harvest Year`, `Cupper.Points` /
  `Overall`, `Moisture` / `Moisture Percentage`), harmonised in clean. The two are
  practically disjoint by grading date, and deduplicated anyway.
- 2018 was written by R: a missing number is `"NA"`, missing text `""`. One row has
  `Total.Cup.Points = 0`; grading dates carry line breaks. `Harvest.Year` is free text
  (`"4T/2010"`, `"Abril - Julio"`, `"2016 / 2017"`).
- Moisture is a fraction in 2018 and a percentage in 2023; `Moisture = 0` (252 rows in
  2018) is physically impossible, so missing.
- Altitude 2018: 182 rows in feet; `altitude_mean_meters` holds garbage (1 m, 190,164 m;
  3,280 = 1,000 m unconverted). Plausible arabica range: 300-3,000 m.
- Countries: 7 CQI names are not in PSD (140 rows). Hawaii and Puerto Rico are PSD's
  `United States` (their production is exactly that); `Cote d?Ivoire` arrives with its
  encoding broken; Myanmar and Mauritius are not in PSD - no context.
- **Target leakage, verified:** `Total Cup Points` is the sum of the ten sensory scores to
  within 0.03 in both snapshots (one exception in 2018). Those ten are never features; a
  test holds it.
- pandera 0.33 (polars backend) with `coerce=True` raises `ColumnNotFoundError` on a
  missing column; `check_contract` checks presence before coercing.

### USDA PSD coffee: `psd_coffee`

`https://apps.fas.usda.gov/psdonline/downloads/psd_coffee_csv.zip`, no key (verified
2026-09-18). One file, `psd_coffee.csv`, long format (`Country_Name`, `Market_Year`,
`Attribute_Description`, `Unit_Description`, `Value`): 87,704 rows, market years
1960-2026. Republished with each circular (June and December); the circular itself is
`https://apps.fas.usda.gov/psdonline/circulars/coffee.pdf` (in the corpus).

### SIAP closing agricultural statistics: `siap_agricola`

Page `https://nube.agricultura.gob.mx/datosAbiertos/Agricola.php` (its links are built by
script; no `.csv` in any href). One municipal CSV a year, 2003-2025:
`https://nube.agricultura.gob.mx/index.php?view=10AE434F-A2158368-A120BC5A-EDF4AFAA&ANIO=<year>`
→ `Cierre_agricola_mun_<year>.csv`, no key. Verified 2026-09-21 (2025) and 2026-09-29
(every year).

- **Latin-1, undeclared**, CRLF. Read as UTF-8 it fails on "Otoño".
- Columns: `Anio, Idestado, Nomestado, Idddr, Nomddr, Idcader, Nomcader, Idmunicipio,
  Nommunicipio, Idciclo, Nomcicloproductivo, Idmodalidad, Nommodalidad, Idunidadmedida,
  Nomunidad, Idcultivo, Nomcultivo, Sembrada, Cosechada, Siniestrada, Volumenproduccion,
  Rendimiento, Preciomediorural, Valorproduccion`. 2025: 35,902 rows, 314 crops.
- **Coffee is `Idcultivo 5710000` "Café cereza", in tonnes**, perennial, rain-fed or
  irrigated: 457-501 rows a year. 2025: 501 rows, 489 municipalities, 15 states,
  1,069,464 t, 8.73 billion pesos.
- **Headers changed twice:** `Precio` until 2020, `Nomcultivo Sin Um` from 2015 to 2020
  (`renamed`). Three cells write a thousands comma (`"10,271"` in 2021, `"3,350.00"` in
  2022) among numbers written bare (`thousands`); two cells of 2009 hold `#¡NUM!` (null).
- A municipality can sit in several CADERs (Ocosingo, Chiapas, in three, across two
  DDRs), and the full key (DDR, CADER, municipality, cycle, modality, crop) repeats twice
  in the file: no uniqueness to demand in raw; clean sums by municipality.
- `Rendimiento` and `Preciomediorural` are empty (415 rows in 2025) only where nothing was
  harvested.
- A closed year is downloaded once; the last is checked every `refresh_hours` (720). The
  host did not accept connections all of 29 September 2026: a source that is down is
  named and the rest go on.

### INEGI borough polygons: `cdmx_boroughs`

Marco Geoestadístico 2020, Mexico City:
`https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/09_ciudaddemexico.zip`,
no key, 83 MB (verified 2026-09-20). Layer `conjunto_de_datos/09mun.shp` (`CVEGEO`,
`CVE_ENT`, `CVE_MUN`, `NOMGEO`), 16 `POLYGON` features, read without unzipping:
`ST_Read('/vsizip/<zip>/conjunto_de_datos/09mun.shp')`.

Traps, all silent:

- **Projection `MEXICO_ITRF_2008_LCC` = EPSG:6372**, metres. Reprojecting to WGS84
  **needs `always_xy := true`**: without it `ST_Transform` does not fail, it returns wrong
  coordinates (the boroughs landed in California).
- **The DBF is Latin-1, undeclared:** `open_options = ['ENCODING=ISO-8859-1']`; read as
  UTF-8 it fails with `Invalid unicode` (or panics in polars, depending on the file).
- `ST_Area_Spheroid` returns NaN on these geometries. Area is computed in the original
  projection (`ST_Area(geom)/1e6`): conformal, not equal-area, so the 16 boroughs add to
  1,486 km² against the 1,495 published (0.6%).
- **`ST_Distance_Sphere` reads the first coordinate as latitude.** With
  `ST_Point(lon, lat)` it does not fail: 0.001° north-south (111 m) comes out as 17.7 m.
  `geo.match_places` builds `ST_Point(lat, lon)`, and a test holds a known distance.
- **DuckDB 1.5.5's spatial LEFT JOIN is broken with thousands of points** (found
  2026-09-27): 64,451 PROFECO points became 67,810 rows - some twice, 327 lost.
  Reproduced with 20,000 random points and two squares; not with 5,000. The inner join
  agrees with `ST_Within` over every pair, so `geo.attribute_points` inner-joins on a row
  index and attaches the result in polars. A regression test lives in `test_geo.py`.
- The `spatial` extension downloads on first use into `~/.duckdb`.
- The 16 boroughs also exist in OSM as `admin_level=6` relations inside
  `area["ISO3166-2"="MX-CMX"]` (not used: INEGI's are the official ones).

### INEGI 2020 Census (ITER): `census_2020`

`https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/iter/iter_09_cpv2020_csv.zip`
(verified 2026-09-29): 175 KB ZIP, 666 rows × 286 columns, UTF-8 with a BOM. A row per
locality plus a total per borough (`LOC 0000`) and for the state (`MUN 000`); `ENTIDAD` +
`MUN` is the polygons' `CVEGEO`. Small localities' figures are withheld as `*`. The
state's 9,209,944 people are exactly the boroughs' sum (the build demands it).
robots.txt allows `/contenidos/programas`; INEGI's terms (free use, citing it).

### World Bank Pink Sheet: `world_bank_prices`

Page `https://www.worldbank.org/en/research/commodity-markets`; the workbook
(`CMO-Historical-Data-Monthly.xlsx`, 587 KB) lives at an address with **an id per
release**, so it is found by its link on the page each time (`link`). Verified
2026-09-25.

- Sheets `Mismatch Details`, `Monthly Prices`, `Monthly Indices`, `Description`,
  `Index Weights`; `Coffee, Arabica` and `Coffee, Robusta` in **$/kg**, months `1960M01`….
  Per its description they are the ICO's other mild Arabicas (New York and
  Bremen/Hamburg, ex-dock) and Robustas (New York and Le Havre/Marseille) group
  indicators, in another unit.
- Read with `pl.read_excel(..., has_header=False, infer_schema_length=0)` (all text; the
  contract types it); an unnamed column becomes `column_<n>`, polars' own convention.
- **`thedocs.worldbank.org` cuts connections** (`WinError 10054`): the download retries a
  transport error up to four times (1, 2, 4 s); an HTTP error is not retried. On
  2026-09-29 it began resetting the TLS handshake of Python's client (curl with the same
  agent got the file): named as failed, not worked around. Checked weekly.
- Licence: none in the file; the World Bank's dataset terms apply (attribution, no
  charging for derivatives).

### ICO indicator prices: `ico_prices`

`https://ico.org/documents/I-CIP.pdf` (linked from the home page; the old
`coffee_prices.asp` answers 500 since the site moved to WordPress). Verified 2026-09-25:
one page, US cents/lb, a row per business day **of the current month only**
(`1-Sep 279.85 380.60 352.86 315.48 172.99`: I-CIP, Colombian Milds, Other Milds,
Brazilian Naturals, Robustas), future days blank, then `Average`, `High`, `Low`,
`DoD Change`, a stray `#REF!` above and a note on group weights since 1 October 2024.

- **The exception to "figures never from a PDF"**, because it checks itself: the reader
  demands that the days average the `Average` row and match `High`/`Low`, and that the
  header is the five columns in order. The ICO averages *unrounded* prices, so the mean
  of the printed days may be half a cent off and the printed average another half:
  tolerances are 0.01 on the average and 0.005 on high and low.
- A page with no days (the 1st of a month) is an empty read, not an error.
- **The history is every download** (`accumulate`): read it before each month ends or
  the days are lost. A day read twice keeps its latest reading.
- Monthly reports: `https://www.ico.org/documents/cy<year>/cmr-<mmyy>-e.pdf`. The ICO's
  Excel files of indicators are asked for at stats@ico.org.

### FRED, pesos per dollar: `fred_usd_mxn`

`https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXMXUS`, no key; robots.txt allows
`/graph/fredgraph.csv` to `*` (Crawl-delay 1; AI crawlers are asked for 30 days).
Verified 2026-09-27: the Federal Reserve's noon buying rate in New York (H.10), daily
since 1993-11-08; 8,575 days, 336 empty (US holidays, an empty field, not "."). The mean
of a month's days is FRED's monthly `EXMXUS` to four decimals in 394 of 394 months, so
the daily series is enough. Chosen over Banxico's FIX because Banxico's API needs a token
and a clean clone should build without one.

### PROFECO, Quién es Quién en los Precios: `profeco_prices`, `profeco_prices_2024`, `profeco_prices_2025`

Page `https://datos.profeco.gob.mx/datos_abiertos/qqp.php`; the links are opaque tokens
(`file.php?t=<hash>`) and the page lists 2025 before 2026, so each file is **found by
its link's text** (`link_text: ^Quien es Quien en los Precios 2026$`). Changing year is
editing the YAML. No robots.txt (404); no licence stated on the page or in the metadata.
Verified 2026-09-27 (2026) and 2026-09-29 (2024, 2025).

- The year in course is a ZIP (`QQP_2026.zip`, 195,609,305 bytes, no `last-modified` or
  `etag`) of fortnightly CSVs (`MM-2026_Q1|Q2.csv`, 150-234 MB each, ~2.5 GB),
  republished monthly with the month before added: year-to-date, so it accumulates
  (2027's file will not carry 2026). Closed years are **RAR 5** (`QQP_2024.rar`,
  `QQP_2025.rar`), opened with libarchive's `bsdtar` (Windows' and macOS's own `tar`;
  Git's GNU tar cannot); files named `MM-YYYY_01.csv` / `_02.csv`.
- Dictionary (`QQP_diccionario_dataset.csv`, UTF-8 with BOM): `producto, presentacion,
  marca, categoria, catalogo, precio, fecha_registro, cadena_comercial, giro,
  nombre_comercial, direccion, estado, municipio, latitud, longitud`.
- **Encoding per file**: UTF-8 with a BOM, except May 2026 (both fortnights) in cp1252
  without one. June 2026 adds three undocumented columns (`folio`, `cv_producto`,
  `cv_marca`); only the fifteen documented are read. May writes dates `dd/mm/yyyy`, the
  rest `yyyy/mm/dd`; every date is checked against its file's fortnight.
- **Coffee is chosen by `producto`** (`Café Soluble`, `Café Tostado y Molido`): `categoria`
  is "Cafe"/"Café" and `catalogo` "Basicos"/"Básicos" depending on the month, and
  searching "caf" catches Cafiaspirina, paracetamol with caffeine and coffee makers. A new
  product in the coffee category is logged.
- **Letters lost as "?"** in June 2026 ("Cl?sico", "Naucalpan de Ju?rez"), and **accents
  missing** in seven states and a chain until November 2025 ("Ciudad de Mexico"): a value
  takes the full spelling only when the same column writes it and exactly one fits.
- A shop can hold two prices for the same product on the same day (24 times): both kept.
- **The borough is the one declared**, not the coordinates': they agree in 114 of 120
  city shops; 2 fall under 200 m from the border and 4 are 1.5-12 km away in another
  borough. INEGI's name is "La Magdalena Contreras" (PROFECO writes "Magdalena
  Contreras").
- Coffee rows: 117,981 (2024), 98,948 (2025), 64,451 (January-July 2026).

### FAOSTAT producer prices: `faostat_prices`

`https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip` (verified
2026-09-29): 11.7 MB ZIP, a 214 MB CSV of 1.3 million rows, UTF-8, every field quoted
except the header. Streamed line by line and kept to **"Coffee, green" (item 656)**:
6,291 rows, 59 countries, 1991-2025, in local currency, the pre-redenomination currency,
US dollars (1991-2024, 52 countries) and an index (2014-2016 = 100); annual values and,
for some countries, months (left out). Licence CC BY 4.0: cite FAO, the dataset and the
date read. The bulk host has no robots.txt (403).

- **Mexico's "green coffee" is the cherry**: FAO's price in pesos is SIAP's rural price
  within five centavos every year 2005-2024 (0.6-0.8% in 2003-2004); every build checks it.
- Series move without a flag: Vietnam halves in 2022 with every flag "official";
  Colombia's is above the port price in 2018-2020.

## APIs

### DENUE (INEGI): `denue_cafes`, `denue_workplaces`

`https://www.inegi.org.mx/servicios/api_denue.html`; a free token (`COFFEE_DENUE_TOKEN`)
that **travels in the URL path**. The registration link on that page is broken (verified
2026-09-20); the working one is
`https://inegi.org.mx/app/desarrolladores/generatoken/Usuarios/token_Verify` and the
token arrives by email. Base: `https://www.inegi.org.mx/app/api/denue/v1/consulta`.

| Method | Path | Used for |
|---|---|---|
| `Cuantificar` | `/Cuantificar/{class}/{area}/{stratum}/{token}` | counts without downloading records; activity 0 = every activity |
| `BuscarAreaAct` | `/BuscarAreaAct/{ent}/{mun}/{loc}/{ageb}/{manz}/{sector}/{subsector}/{rama}/{clase}/{nombre}/{desde}/{hasta}/{id}/{token}` | **the inventory**: class + area, paged |
| `Buscar` | `/Buscar/{condition}/{lat},{lon}/{metres}/{token}` | radius ≤ 5,000 m |
| `Nombre`, `BuscarEntidad`, `BuscarAreaActEstr`, `Ficha` | | by name, by entity, with stratum, detail by id |

- **SCIAN `722515`** = "Cafeterías, fuentes de sodas, neverías, refresquerías y
  similares": wider than coffee (by name: coffee 38%, unclassified 28%, juices 24%, soda
  fountains 4%, school tuck shops 3%, ice cream 2%). Classified by name in cleaning, never
  filtered.
- Mexico City: **9,860**; Cuauhtémoc 1,577, Benito Juárez 882 (2026-09-20). Pages of 100,
  34 fields per establishment, every row with a coordinate, all 16 boroughs.
- **`BuscarAreaAct` does not return `Municipio`** (nor `Localidad` nor `Ageb`: empty). The
  borough travels in **`AreaGeo`** (`090050001` = entity 09 + municipality 005 +
  locality 0001): its first five characters are the polygons' `CVEGEO`. Useful fields:
  `Id`, `Nombre`, `Clase_actividad`, `CLASE_ACTIVIDAD_ID`, `AreaGeo`, `Latitud`,
  `Longitud`, `Estrato`, `Colonia`, `CP`, `Fecha_Alta` (the register's edition,
  `"2024-11"`; two written `"2013 07"`).
- **Rows come in a different order on every call**: sorted before storing.
- `denue_workplaces`: `Cuantificar` with activity 0, per borough and staff stratum
  (16 × 7 = 112 requests). The strata match the records' own `Estrato` (1 = 0-5 people …
  6 = 101-250, 7 = 251+), verified 2026-09-29. Each SCIAN level repeats the total: sum at
  the sector level.

### USDA FAS Open Data: `fas_psd_coffee`

**The documented address is dead**: `https://apps.fas.usda.gov/OpenData/api/...` with the
`API_KEY` header answers 500 with any key, even an invented one; without the header, 403
(verified 2026-09-20). The live API is **`https://api.fas.usda.gov/api/`** with the header
**`X-Api-Key`** (`?api_key=` also works; the header keeps the key out of URLs and logs).
Free key from `https://apps.fas.usda.gov/opendataweb/`; `COFFEE_USDA_FAS_API_KEY`.

- Working endpoints: `psd/commodities`, `psd/countries`, `psd/regions`,
  `psd/commodityAttributes`, `psd/unitsOfMeasure`, `gats/countries`, `esr/commodities`,
  and the data at `psd/commodity/{code}/country/{cc}/year/{year}`,
  `.../world/year/{year}` and `.../country/all/year/{year}`. **Coffee = `0711100`**
  ("Coffee, Green").
- **1,000 requests an hour per key** (`x-ratelimit-limit`); a full pull is ~72 (68 years
  + 4 catalogues).
- Answers ids, not names (`commodityCode, countryCode, marketYear` (text),
  `calendarYear, month, attributeId, unitId, value`); the catalogues (63 commodities, 85
  attributes, 42 units, 251 countries) are stored beside the rows. `unitDescription` is
  space-padded (`"(1000 60 KG BAGS)   "`). A year without data answers `[]` with 200;
  1960 is coffee's first.
- **Reconciled with the PSD file: 87,704 rows in both, the same keys, no value
  different** (2026-09-21), and again on every build. The file stays the source of
  `market_context` (no key, one request).

### OpenStreetMap / Overpass: `osm_places`

`https://overpass-api.de/api/interpreter`, no key, GET with `?data=<Overpass QL>`. ODbL
allows storing and redistributing (Places does not), so it is the open register of
places. Verified 2026-09-20.

- One query per run, `amenity~^(cafe|ice_cream)$`: 1,357 elements (1,125 cafe + 232
  ice_cream). Cafes: 1,031 nodes + 94 ways + 0 relations; 1,025 named, 407 with
  `cuisine`, 163 with `brand`.
- **Ways have no coordinate of their own**: without `out center`, 94 of 1,125 are lost.
  Relations are in the union even though there are none today.
- **Overpass reports its failures inside a 200**: a `remark` (timeout, out of memory,
  syntax) in the body, and a short inventory looks complete. The extractor raises on it.
- The query's `[timeout:N]` stays below the client's read timeout (60 s), so the server's
  explanation arrives instead of a bare client timeout.
- A volunteer instance: it answered 429 twice during verification (the client's retries
  absorbed it).
- `osm3s.timestamp_osm_base` changes every minute: stored, it would break the sha256
  deduplication. The sorted elements, the query and the ODbL notice are stored; the date
  is in the manifest.
- Coverage is uneven: OSM has 11.4% of DENUE's places, 25% in Cuauhtémoc against ~3% in
  Iztapalapa. Both registers are kept, unmerged, with a `source` column.

## Roasters' shops: `roaster_catalogs`

robots.txt allows the project's agent on all four; none asks for a `Crawl-delay`
(verified 2026-09-21). Requests at least 3 s apart; one raw document for all shops.

| Shop | Platform | Catalogue | Coffee / all | Where the attributes are |
|---|---|---|---|---|
| Almanegra `almanegra.cafe` | Shopify | `/products.json` (2 pages of 250) | 130 / 284, `product_type: Coffee` | a sheet in `body_html`, a line per label |
| Buna `buna.mx` | Shopify | `/products.json` | 23 / 74, tag `Café` (`product_type` mostly empty) | **only on the product page**: `<h5>Región</h5><p>…</p>` (Región, Altura, Variedades, Proceso) |
| Café con Jiribilla `cafeconjiribilla.com` | Shopify | `/products.json` | 3 / 3 (sells only coffee, untagged) | a clean sheet in `body_html` (Origen, Productor, Predio, Variedad, Proceso) |
| Cucurucho `cucuruchocafe.com` | Squarespace | `/tienda?format=json`, unpaged | 12 / 23 (tote bags, shirts, a jacket, subscriptions and a cup excluded) | prose only; the **title** carries region and notes ("Chiapas- Caramelo, avellana y chocolate") |

- Almanegra: label variants (`Región`/`Region`, `Productor`/`Productores`/`Productora(s)`,
  `Proceso`/`Procesos`, `Varietal(es)`), altitude as a range (`1,200 - 1,700 msnm`,
  `2,000 a 2,400 msnm`), and **`Tostado:` is the roaster's name, not the roast level**.
  Labels glued to the value before them ("YemenRegión: ..."): a value ends at the next
  known label. 2 products carry a `Puntaje SCA`.
- Buna: the weight is in the variant's title (`340 gr / Molido medio`) with `grams: 0`;
  blends repeat the headings per component (Guarumbo = 2 arabicas + 1 robusta), hence a
  row per origin. Its pages change their whitespace between requests: stored without it.
- Jiribilla contradicts itself: tag `natural`, sheet `Proceso: Lavado`.
- Cucurucho: the price is in `variants[].priceMoney.value` (the product's `priceMoney` is
  0.00); the variant's `weight` is 0.0 with an unverified unit (not used); `body` is
  `None` and the text is in `excerpt`. How Squarespace paginates is not verified: if a
  `pagination` object appears, the reader refuses rather than read one page and stop.
- **The platform's weight lies** in 77 offers (Jiribilla "1 kg" at 250 g; Almanegra
  "1.25 kg" at 313 g): the size comes only from titles, else null. Three prices copied
  from another size are kept and flagged `price_outlier` (>3× or <1/3 of the product's
  median per kg).
- Currency: Buna's and Squarespace's pages declare `"currencyCode":"MXN"`; Shopify's
  `products.json` does not.
- Vocabularies are aligned with the other tables: countries as PSD's, states as SIAP's
  ("México" = Estado de México), processes and varieties as the CQI's; "heirloom" stays
  heirloom. Unrecognised values are logged and left null (rules open, not closed: shops
  change weekly).
- 2026-09-21: 167 coffees in 527 offers (a product in one size), 6 MB of raw.

## Documents (the corpus)

17 documents indexed, each downloaded and its text read (2026-09-21 to 23); 5 more
fetched on 2026-09-29 and waiting to be chunked, indexed and measured. None is committed:
the ones behind a refusal are fetched by hand into `data/coffee/inbox/documents/`.

| Topic | Document | Access | Form | Licence |
|---|---|---|---|---|
| varieties | WCR arabica catalogue (`.../catalogEnglish/Combined/arabica-1769602235.pdf`) | automatic | PDF, 86 p | none stated; cite |
| varieties | WCR robusta catalogue (`.../robusta-1769602236.pdf`) | automatic | PDF, 67 p | none stated; cite |
| cultivation, processing | FAO, *Arabica coffee manual for Lao PDR* (2005), copy at laocoffee.org; official HTML at `fao.org/4/ae939e` | automatic | PDF, 72 p | © FAO, non-commercial with citation |
| wet processing | Frontiers in Microbiology 2019, PMC6863779 | Europe PMC API | JATS | CC BY |
| roasting | Molecules 2024, roast level and aroma, PMC11477549 | Europe PMC API | JATS | CC BY 4.0 |
| brewing | J. Math. Industry 2016, extraction kinetics, PMC4986356 | Europe PMC API | JATS | CC BY 4.0 |
| market | ICO Coffee Market Report (`https://www.ico.org/documents/cy2025-26/cmr-0826-e.pdf`) | automatic | PDF, 17 p | none stated; cite |
| market, sustainability | ICO Coffee Development Report 2022-23 | automatic | PDF, 118 p, 35 MB | © 2024 ICO; internal use, cite |
| market | USDA FAS *Coffee: World Markets and Trade* | automatic | PDF, 10 p | public domain |
| cupping | SCA CVA forms; SCA-102, 103, 104, 105-2025 (`https://sca.coffee/value-assessment`) | **by hand** (direct links 404) | PDF, encrypted with permissions | © SCA, reproduction allowed |
| flavour, chemistry | MDPI Beverages 2020, *Coffee Flavor: A Review* | **by hand** (403) | PDF, 25 p | CC BY |
| roasting | MDPI Beverages 2020, *Roasting Conditions and Coffee Flavor* | **by hand** (403) | PDF, 14 p | CC BY |
| processing, aroma | IJFST 2023, postharvest processing and aroma | **by hand** (403) | PDF, 21 p | CC BY |
| brewing (waiting) | Molecules 2025 (PMC12565998), Antioxidants 2023 (PMC10812495), Scientific Reports 2024 (PMC11586412) | Europe PMC API | JATS | CC BY 4.0 |
| market, cultivation (waiting) | USDA GAIN *Mexico: Coffee Annual* 2025 and 2026 | GAIN API, download by file name | PDF, 8 and 15 p | public domain |

- **PubMed Central answers automated clients with a reCAPTCHA.** It is not evaded: Europe
  PMC has a documented API (`/webservices/rest/<PMCID>/fullTextXML`), which also serves
  the text by section.
- **MDPI and Oxford Academic answer 403** to automated clients, even for CC BY content;
  the SCA's direct links 404 and `sca.coffee/research/protocols-best-practices` too. The
  2004 cupping protocol was replaced by the Coffee Value Assessment (SCA-102 to 105).
- The GAIN site's own links answer 403 to a client; its API's download by file name
  serves the PDF (no robots.txt, 404).
- Extraction traps: typographic ligatures (`Coﬀee`, normalised with NFKC), dotted form
  lines (`Name . . . . .`), and PDFs encrypted with permissions (an empty password opens
  them; pypdf needs `cryptography`). The CVA forms are score sheets - vocabulary, not
  explanation.
- Known gap: a practical roasting guide (curves, first crack, Agtron). Not added ahead of
  need: it comes in if retrieval shows roasting questions failing.

## Candidates: verified, rejected or waiting

| Source | State | Why |
|---|---|---|
| Google Places | **never persisted** | its terms restrict storing and caching |
| Yahoo `KC=F` (Coffee C futures) | rejected | unofficial endpoint, personal-use terms, and a future, not the physical price |
| FRED `PCOFFOTMUSDM`, `PCOFFROBUSDM` | verified alternative, not chosen | the IMF's monthly other milds and robustas (¢/lb, 1992→); the World Bank's go back to 1960 |
| Banxico SIE (FIX) | not chosen | its API needs a token; FRED does not |
| Metro ridership, `datos.cdmx.gob.mx` | **in progress** | CC-BY-4.0-ESP; robots.txt forbids `/api/` and asks `Crawl-delay: 10` (files only, ten seconds apart: `SourceConfig.crawl_delay`); the 195 station-lines of the daily ridership match the GTFS; names double-encoded in 2021-2023 (UTF-8 read as Latin-1 and encoded again); ~62,000 days at zero (closed stations); in December 2020 "Deportivo Oceanía" also appears as "Oceanía" - indistinguishable rows, dropped |
| ENIGH 2024 microdata (INEGI) | verified, not read | 200, 54 MB ZIP; the household spending that would fill the consumption gap; its geographic representativeness (probably state, not borough) is still to check |
| Cup of Excellence Mexico 2012-2026 | verified, terms to check | HTML tables (score, farm, process, variety, region); robots.txt allows them |
| INEGI INPC (consumer prices) | needs a token | the indicators API takes its own token, not DENUE's; would deflate SIAP's and PROFECO's nominal prices |
| SCA Specialty Coffee Transaction Guide | needs a person | a form asking for personal data |
| Open-Meteo | not used | its robots.txt forbids this project's agent |
| Censo Agropecuario 2022 | not found | the open-data link tried was not valid |
