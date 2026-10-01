"""Shared fixtures. HTTP is replayed from recorded payloads in tests/fixtures/.

`httpx` and the ETL package are imported inside the fixtures that need them, so the
serving tests can run in an environment that installed only the `serving` extra. That
is how CI proves the API's dependency list is complete instead of relying on packages
another pipeline happens to pull in.
"""

import csv
import functools
import importlib.util
import io
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

import domains.coffee
from domains.coffee.adapter import CoffeeAdapter
from domains.coffee.config import CoffeeConfig, CoffeeCredentials
from mlops_core.config import Settings
from tests.fakes import RecordedServer, fas_recording, without_rate_limits
from tests.files import pdf, xlsx

FIXTURES = Path(__file__).parent / "fixtures"

# Kaggle answers with a 302 to a short-lived signed URL like this one.
SIGNED_URL = "https://storage.googleapis.com/kaggle-data-sets/archive.zip?X-Goog-Signature=abc123"

# The World Bank's page links its workbook under an id per release: the page is served,
# and so is the file at the address it links - among other links, one of them relative.
WORLD_BANK_FILE = (
    "https://thedocs.worldbank.org/en/doc/r2026/related/CMO-Historical-Data-Monthly.xlsx"
)
WORLD_BANK_PAGE = (
    '<a href="/en/research/pink-sheet">Pink Sheet</a>'
    f'<a href="{WORLD_BANK_FILE.replace("Monthly", "Annual")}">Annual</a>'
    f'<a href="{WORLD_BANK_FILE}">Monthly</a>'
).encode()
# The Pink Sheet's layout: title rows, names, units, then months - three of them, the
# last one also on the ICO's page below, so the two publishers can be compared.
WORLD_BANK_SHEET = [
    ["World Bank Commodity Price Data (The Pink Sheet)"],
    ["monthly prices in nominal US dollars, 1960 to present"],
    ["(monthly series are available only in nominal US dollars)"],
    ["Updated on September 02, 2026"],
    [None, "Crude oil, average", "Coffee, Arabica", "Coffee, Robusta"],
    [None, "($/bbl)", "($/kg)", "($/kg)"],
    ["2026M07", 69.9, 7.91, 4.07],
    ["2026M08", 68.2, 7.97, 3.98],
    ["2026M09", 67.5, 7.29, 3.70],
]
# The ICO's page as pypdf reads it: a stray spreadsheet error, a header wrapped over
# three lines, days to come left blank, and the summary rows that check the rest.
ICO_PAGE = [
    "#REF!",
    "ICO Indicator Prices - September 2026 (I-CIP)",
    "In US cents/lb",
    "I-CIP Colombian",
    " Milds Other Milds Brazilian",
    " Naturals Robustas",
    "1-Sep 279.85 380.60 352.86 315.48 172.99",
    "2-Sep 270.47 368.53 340.80 303.07 168.68",
    "3-Sep 268.38 366.51 338.76 300.83 166.66",
    "4-Sep     ",
    "Average 272.90 371.88 344.14 306.46 169.44",
    "High 279.85 380.60 352.86 315.48 172.99",
    "Low 268.38 366.51 338.76 300.83 166.66",
    "DoD Change -0.8% -0.5% -0.6% -0.7% -1.2%",
    "\N{COPYRIGHT SIGN} International Coffee Organization",
]


# PROFECO's page names its files only in its links' text; the addresses are tokens, and
# last year's is listed first.
PROFECO_FILE = "https://datos.profeco.gob.mx/datos_abiertos/file.php?t=9d62"
# The closed years' archives, found by their links' text like the year in course.
PROFECO_CLOSED = {
    "profeco_prices_2025": "https://datos.profeco.gob.mx/datos_abiertos/file.php?t=b954",
    "profeco_prices_2024": "https://datos.profeco.gob.mx/datos_abiertos/file.php?t=2de3",
}
PROFECO_PAGE = (
    b'<a href="https://www.gob.mx/profeco">PROFECO</a><a href="index.php"><img src="l.png"></a>'
    b'<a href="file.php?t=b954">\n  Quien es Quien en los Precios 2025</a>'
    b'<a href="file.php?t=9d62"><span>Quien es Quien en los</span> Precios 2026</a>'
    b'<a href="file.php?t=2de3">Quien es Quien en los Precios 2024</a>'
    b'<a href="file.php?t=42ed">Metadatos dataset</a>'
)
# Stores: chain, kind, name, address, state, municipality, latitude, longitude. The two
# in the city stand where the boundary fixture puts Miguel Hidalgo and La Magdalena
# Contreras; the market says Miguel Hidalgo and stands in the other.
POLANCO = ("Wal-mart", "Supermercado / Tienda de Autoservicio", "Walmart Sucursal Polanco",
           "Ejercito Nacional 843. Cp. 11520", "Ciudad de México", "Miguel Hidalgo",
           "19.45", "-99.15")  # fmt: skip
CONTRERAS = ("Soriana Super", "Supermercado / Tienda de Autoservicio",
             "Soriana Super Sucursal Contreras", "San Jeronimo 630. Cp. 10200",
             "Ciudad de México", "Magdalena Contreras", "19.3048187", "-99.1022689")  # fmt: skip
MARKET = ("Mercado Publico", "Mercados", "Mercado Tacuba", "Calz. Mexico Tacuba s/n",
          "Ciudad de México", "Miguel Hidalgo", "19.3048", "-99.1023")  # fmt: skip
XALAPA = ("Chedraui", "Supermercado / Tienda de Autoservicio", "Chedraui Sucursal Xalapa",
          "Av. Lazaro Cardenas 300", "Veracruz", "Xalapa", "19.54", "-96.91")  # fmt: skip
QQP_COLUMNS = ["producto", "presentacion", "marca", "categoria", "catalogo", "precio",
               "fecha_registro", "cadena_comercial", "giro", "nombre_comercial", "direccion",
               "estado", "municipio", "latitud", "longitud"]  # fmt: skip
INSTANT, GROUND = "Café Soluble", "Café Tostado y Molido"


def shelf(product: str, presentation: str, brand: str, price: str, day: str,
          store: tuple[str, ...], category: str = "Café") -> list[str]:  # fmt: skip
    """One row of a fortnight's file, in the dictionary's column order."""
    return [product, presentation, brand, category, "Básicos", price, day, *store]


# Three fortnights of 2026, each one of the ways the real files differ: May's cp1252
# with day-first dates, June's with three undocumented columns and letters lost to "?",
# July's with a coffee product nobody listed. Beside the coffee, what is not coffee.
QQP_FORTNIGHTS: dict[str, tuple[str, list[str], list[list[str]]]] = {
    "QQP_2026/05-2026_Q1.csv": ("cp1252", QQP_COLUMNS, [
        shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "110", "04/05/2026", POLANCO),
        shelf(GROUND, "Bolsa 400 Gr. Mezclado con Caramelo", "Legal", "95.9", "04/05/2026",
              POLANCO),
        shelf(INSTANT, "Frasco 170 Gr. Sin Cafeína. Descafeinado", "Nescafé. Decaf", "160",
              "12/05/2026", XALAPA),
        shelf("Cafeteras", "Eléctrica 12 Tazas", "Oster", "899", "04/05/2026", POLANCO,
              category="Aparatos Eléctricos"),
    ]),
    "QQP_2026/06-2026_Q1.csv": ("utf-8-sig", [*QQP_COLUMNS, "folio", "cv_producto",
                                              "cv_marca"], [
        [*shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Cl?sico", "112", "2026/06/03", POLANCO),
         "1", "11", "7"],
        # The same price twice once the letter is back: one row.
        [*shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "112", "2026/06/03", POLANCO),
         "2", "11", "7"],
        # Two prices on one shelf on one day: both kept.
        [*shelf(GROUND, "Bolsa 400 Gr.", "Internacional Americano", "160", "2026/06/05",
                CONTRERAS), "3", "12", "8"],
        [*shelf(GROUND, "Bolsa 400 Gr.", "Internacional Americano", "162", "2026/06/05",
                CONTRERAS), "4", "12", "8"],
        [*shelf("Leche", "Caja 1 Lt.", "Lala", "28", "2026/06/05", CONTRERAS,
                category="Leche"), "5", "13", "9"],
    ]),
    "QQP_2026/07-2026_Q2.csv": ("utf-8-sig", QQP_COLUMNS, [
        shelf(INSTANT, "Frasco 200 Gr.", "Nescafé. Clásico", "190", "2026/07/20", POLANCO),
        shelf(GROUND, "Bolsa 400 Gr.", "Internacional Americano", "158", "2026/07/21",
              MARKET),
        shelf(GROUND, "Bolsa 400 Gr.", "Internacional Americano", "165", "2026/07/22",
              XALAPA),
        shelf("Café en Cápsulas", "Caja 10 Pzas.", "Dolce Gusto", "150", "2026/07/20",
              POLANCO),
    ]),
}  # fmt: skip


# The closed years name a fortnight `_01` and `_02`, not `_Q1` and `_Q2`.
QQP_CLOSED: dict[str, dict[str, tuple[str, list[str], list[list[str]]]]] = {
    "2025": {"QQP_2025/12-2025_02.csv": ("utf-8-sig", QQP_COLUMNS, [
        shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "105", "2025/12/17", POLANCO),
        shelf(GROUND, "Bolsa 400 Gr.", "Internacional Americano", "150", "2025/12/18",
              CONTRERAS),
    ])},
    "2024": {"QQP_2024/03-2024_01.csv": ("utf-8-sig", QQP_COLUMNS, [
        shelf(INSTANT, "Frasco 120 Gr.", "Nescafé. Clásico", "95", "2024/03/05", POLANCO),
    ])},
}  # fmt: skip


def qqp_archive(fortnights: dict[str, tuple[str, list[str], list[list[str]]]]) -> bytes:
    """A year of PROFECO's survey: a folder of fortnightly CSVs in one ZIP, each file
    in its own character set, CRLF, every field quoted where it needs to be."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for member, (encoding, columns, rows) in fortnights.items():
            text = io.StringIO()
            csv.writer(text, lineterminator="\r\n").writerows([columns, *rows])
            archive.writestr(member, text.getvalue().encode(encoding))
    return buffer.getvalue()


# INEGI's census shaped as it downloads: a BOM, a ZIP, a folder. Three of the boundary
# fixture's boroughs, the state's total as their sum, and a small locality whose figures
# are withheld.
CENSUS_MEMBER = "iter_09_cpv2020/conjunto_de_datos/conjunto_de_datos_iter_09CSV20.csv"
CENSUS_ROWS = [
    "ENTIDAD,NOM_ENT,MUN,NOM_MUN,LOC,NOM_LOC,POBTOT,P_18YMAS,TVIVHAB,GRAPROES,PEA",
    "09,Ciudad de México,000,Total de la entidad Ciudad de México,0000,Total de la Entidad,"
    "1207976,979460,411528,12.70,678601",
    "09,Ciudad de México,008,La Magdalena Contreras,0000,Total del Municipio,"
    "247622,190500,68107,11.20,137000",
    "09,Ciudad de México,008,La Magdalena Contreras,0021,Tierra Colorada,12,*,4,*,*",
    "09,Ciudad de México,015,Cuauhtémoc,0000,Total del Municipio,545884,452000,196593,12.41,300000",
    "09,Ciudad de México,016,Miguel Hidalgo,0000,Total del Municipio,"
    "414470,336960,146828,14.20,241601",
]


def census_archive(rows: list[str] = CENSUS_ROWS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CENSUS_MEMBER, "﻿" + "\r\n".join(rows) + "\r\n")
    return buffer.getvalue()


# INEGI's 2025 Intercensal Survey shaped as it downloads: Windows-1252, a CSV in a ZIP, a
# row per area and estimator. Three of the boundary fixture's boroughs and the state as
# their sum; a locality and another state's municipality, which the profile leaves out;
# and an indicator whose sample was too small in one borough ("MI").
SURVEY_MEMBER = "conjunto_de_datos/conjunto_datos_eic2025_105.csv"
SURVEY_INDICATORS = [
    "POBTOT", "MEDIANA_POBTOT", "INDICE_ENV", "RAZON_DEP_TOT", "PCN_PRESOE20", "GRAPROES",
    "PCN_P15YM_ES", "PCN_PEA", "PCN_PDESOCUP", "PCN_POCUP_ASA", "PCN_POCUP_CPRO",
    "PCN_PSINDER", "PCN_POCUP_OENT", "PCN_POCUP_MET", "PCN_POCUP_2HOR", "PCN_POCUP_2HYM",
    "TOTHOG", "HOGJEF_F", "VIVPARHAB", "PROM_OCUP", "PRO_OCUP_C", "PCN_VPH_PROPIA",
    "PCN_VPH_ALQUI", "PCN_VPH_INTER", "PCN_VPH_AUTOM", "PCN_VPH_PC",
]  # fmt: skip
SURVEY_COUNTS = {"POBTOT", "TOTHOG", "HOGJEF_F", "VIVPARHAB"}
SURVEY_ESTIMATOR_NAMES = [
    "Valor", "Error estándar", "Límite inferior de confianza",
    "Límite superior de confianza", "Coeficiente de variación",
]  # fmt: skip
# (CVEGEO, municipality, its name, locality, people, median age, % rented)
SURVEY_AREAS = [
    ("090080000", "008", "La Magdalena Contreras", "0000", 250000, 36.5, 20.0),
    ("090150000", "015", "Cuauhtémoc", "0000", 505637, 37.55, 42.15),
    ("090160000", "016", "Miguel Hidalgo", "0000", 410000, 39.0, 42.31),
    ("090150001", "015", "Cuauhtémoc", "0001", 505637, 37.55, 42.15),  # the locality
    ("140390000", "039", "Guadalajara", "0000", 1400000, 33.0, 30.0),  # another state
]


def survey_row(cvegeo: str, mun: str, name: str, loc: str, people: float, age: float,
               rented: float, estimator: str) -> list[str]:  # fmt: skip
    """One area's figures under one estimator: the value, an interval 5% either side, a
    3% standard error and a coefficient of 3."""
    values = {column: (round(people / 2.5) if column in SURVEY_COUNTS else 10.0)
              for column in SURVEY_INDICATORS}  # fmt: skip
    values.update({"POBTOT": people, "MEDIANA_POBTOT": age, "PCN_VPH_ALQUI": rented})
    scale = {"Valor": 1.0, "Error estándar": 0.03, "Límite inferior de confianza": 0.95,
             "Límite superior de confianza": 1.05}  # fmt: skip
    figures = [
        "3.0" if estimator.startswith("Coef") else f"{values[c] * scale[estimator]:.10g}"
        for c in SURVEY_INDICATORS
    ]
    if mun == "008":  # too few people sampled in this borough had moved since 2020
        figures[SURVEY_INDICATORS.index("PCN_PRESOE20")] = "MI"
    return [cvegeo, cvegeo[:2], "Ciudad de México", mun, name, loc, name, estimator, *figures]


def survey_archive(
    areas: list[tuple[str, str, str, str, int, float, float]] = SURVEY_AREAS,
) -> bytes:
    boroughs = [a for a in areas if a[0].startswith("09") and a[3] == "0000"]
    state = ("090000000", "000", "Total de la entidad", "0000",
             sum(a[4] for a in boroughs), 37.0, 35.0)  # fmt: skip
    rows = [survey_row(*area, estimator)
            for area in [state, *areas] for estimator in SURVEY_ESTIMATOR_NAMES]  # fmt: skip
    header = ["CVEGEO", "CVE_ENT", "NOM_ENT", "CVE_MUN", "NOM_MUN", "CVE_LOC", "NOM_LOC",
              "ESTIMADOR", *SURVEY_INDICATORS]  # fmt: skip
    text = io.StringIO()
    csv.writer(text, lineterminator="\r\n").writerows([header, *rows])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(SURVEY_MEMBER, text.getvalue().encode("cp1252"))
    return buffer.getvalue()


# INEGI's geostatistical framework as the tests download it: the boundary fixture's
# sixteen boroughs (a 4x4 grid of squares in the layer's projection) and, written here,
# the urban AGEBs: 2,431 cells of a 50x50 grid over the same squares - as many as the real
# layer has, so the config's count holds - each keyed by the borough its corner is in.
AGEB_CELL = 0.0104  # degrees; the boroughs' squares are 0.13
AGEB_ORIGIN = (-99.45, 19.05)  # the grid's south-west corner, as the boroughs'


def ageb_code(latitude: float, longitude: float) -> str:
    """The AGEB the written framework draws around a point: "09" + borough + "0001" + n."""
    col = int((longitude - AGEB_ORIGIN[0]) / AGEB_CELL)
    row = int((latitude - AGEB_ORIGIN[1]) / AGEB_CELL)
    borough = 2 + int(row * AGEB_CELL / 0.13) * 4 + int(col * AGEB_CELL / 0.13)
    return f"09{borough:03d}0001{row * 50 + col:04d}"


@functools.cache
def framework_archive() -> bytes:
    if importlib.util.find_spec("duckdb") is None:
        # The API's isolated tests run without DuckDB and read no map: the boroughs'
        # fixture alone stands in for the framework.
        return (FIXTURES / "cdmx_boroughs_sample.zip").read_bytes()
    from mlops_core.data.geo import spatial_connection

    west, south = AGEB_ORIGIN
    query = f"""
        COPY (
            SELECT '09' || lpad(CAST(CAST(2 + floor(row * {AGEB_CELL} / 0.13) * 4
                                           + floor(col * {AGEB_CELL} / 0.13) AS INTEGER)
                                      AS VARCHAR), 3, '0')
                       || '0001' || lpad(CAST(n AS VARCHAR), 4, '0') AS CVEGEO,
                   lpad(CAST(n AS VARCHAR), 4, '0') AS CVE_AGEB,
                   ST_Transform(ST_MakeEnvelope({west} + col * {AGEB_CELL},
                                                {south} + row * {AGEB_CELL},
                                                {west} + (col + 1) * {AGEB_CELL},
                                                {south} + (row + 1) * {AGEB_CELL}),
                                'EPSG:4326', 'EPSG:6372', always_xy := true) AS geom
            FROM (SELECT n, n % 50 AS col, n // 50 AS row FROM range(2431) t(n))
        ) TO '{{target}}' WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile')
    """  # fmt: skip
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as folder, closing(spatial_connection()) as con:
        con.execute(query.format(target=(Path(folder) / "09a.shp").as_posix()))
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            with zipfile.ZipFile(FIXTURES / "cdmx_boroughs_sample.zip") as boroughs:
                for member in boroughs.namelist():
                    archive.writestr(member, boroughs.read(member))
            for path in sorted(Path(folder).iterdir()):
                archive.write(path, f"conjunto_de_datos/{path.name}")
    return buffer.getvalue()


# The 2020 Census by urban AGEB as it downloads: a BOM, a ZIP, a folder; totals for the
# state, a borough and a locality, then each AGEB's total row and its blocks. Six AGEBs of
# Cuauhtémoc's square (Balderas', the Zócalo's and the Metrobús' among them), one where
# nobody lives, one the framework does not draw, and a withheld figure.
CENSUS_AGEB_MEMBER = (
    "ageb_mza_urbana_09_cpv2020/conjunto_de_datos/conjunto_de_datos_ageb_urbana_09_cpv2020.csv"
)
CENSUS_AGEB_HEADER = ("ENTIDAD,NOM_ENT,MUN,NOM_MUN,LOC,NOM_LOC,AGEB,MZA,POBTOT,P_18YMAS,TVIVHAB,"
                      "GRAPROES,PEA,POB65_MAS,VPH_INTER,VPH_AUTOM,VPH_PC")  # fmt: skip


def ageb_row(code: str, name: str, block: str, people: int, schooling: str = "12.5") -> str:
    mun, loc, ageb = code[2:5], code[5:9], code[9:]
    return (f"09,Ciudad de México,{mun},Cuauhtémoc,{loc},{name},{ageb},{block},{people},"
            f"{people - 50},{people // 3},{schooling},{people // 2},{people // 10},"
            f"{people // 4},{people // 6},{people // 5}")  # fmt: skip


BALDERAS_ZONE = ageb_code(19.50, -99.25)
CENSUS_AGEB_ROWS = [
    CENSUS_AGEB_HEADER,
    "09,Ciudad de México,000,Total de la entidad,0000,Total de la entidad,0000,000,"
    "9209944,7000000,2700000,11.5,5000000,1300000,2000000,1300000,1100000",
    "09,Ciudad de México,015,Cuauhtémoc,0000,Total del municipio,0000,000,"
    "545884,452000,196593,12.4,300000,70000,150000,60000,90000",
    "09,Ciudad de México,015,Cuauhtémoc,0001,Total de la localidad urbana,0000,000,"
    "545884,452000,196593,12.4,300000,70000,150000,60000,90000",
    ageb_row(BALDERAS_ZONE, "Total AGEB urbana", "000", 3000),
    ageb_row(BALDERAS_ZONE, "Cuauhtémoc", "001", 1500),  # a block: not read
    ageb_row(ageb_code(19.51, -99.24), "Total AGEB urbana", "000", 1500),  # the Zócalo
    ageb_row(ageb_code(19.505, -99.255), "Total AGEB urbana", "000", 2400),  # Metrobús
    ageb_row(ageb_code(19.49, -99.27), "Total AGEB urbana", "000", 900, schooling="*"),
    ageb_row(ageb_code(19.48, -99.28), "Total AGEB urbana", "000", 0, schooling="0"),
    ageb_row(ageb_code(19.47, -99.29), "Total AGEB urbana", "000", 4100),
    ageb_row("0901500019999", "Total AGEB urbana", "000", 75),  # no polygon draws it
]


# ENIGH, as INEGI writes it: a household a row (its weight, place, stratum, sampling unit,
# size and income) and a purchase a row, blanks as a space. Eight households in three
# states, two strata and four sampling units; their coffee: instant and ground bought,
# coffee from a household's own harvest, a gift (left out) and a purchase that is not
# coffee (not read).
ENIGH_HOUSEHOLDS = [
    "folioviv,foliohog,ubica_geo,tam_loc,est_dis,upm,factor,tot_integ,ing_cor",
    "0900000101,1,09015,1,001,0000001,300,3,90000.00",
    "0900000102,1,09015,1,001,0000001,300,2,30000.00",
    "0900000201,1,09003,1,001,0000002,200,4,150000.00",
    "0900000202,1,09003,1,001,0000002,200,1,12000.00",
    "0700000101,1,07059,3,002,0000003,500,5,15000.00",
    "0700000102,1,07059,3,002,0000003,500,3,24000.00",
    "1500000101,1,15057,1,002,0000004,800,4,60000.00",
    "1500000101,2,15057,1,002,0000004,800,2,45000.00",
]
ENIGH_SPENDING = [
    "folioviv,foliohog,clave,tipo_gasto,mes_dia,gasto,gasto_tri,gas_nm_tri,entidad,factor",
    "0900000101,1,012201,G1,1031,30,385.71, ,09,300",
    "0900000101,1,012202,G1,1101,40,514.28, ,09,300",
    "0900000201,1,012202,G1,1101,90,1157.14, ,09,200",
    "0900000201,1,012202,G1,1104,10,128.57, ,09,200",  # a second bag the same week
    "0700000101,1,012201,G3,1102, , ,257.14,07,500",  # from its own harvest
    "0700000102,1,012201,G1,1102,15,192.85, ,07,500",
    "1500000101,2,012203,G5,1103, , ,64.28,15,800",  # a gift: neither paid nor grown
    "1500000101,1,011131,G1,1031,13,167.14, ,15,800",  # bread: not coffee
]


def html_table(*rows: list[str]) -> str:
    cells = ["<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows]
    return "<table>" + "".join(cells) + "</table>"


COE_JUDGES = html_table(["Name", "Country", "Company"], ["Head Judge"], ["E. M.", "Nicaragua", ""])


def coe_page(year: int) -> bytes:
    """A year of Cup of Excellence Mexico as the site lays it out: before 2018 a results
    table and an auction table by lot number, with boxes for a size and a row of totals;
    from 2018 scores, weights, varieties and processes on both. The last year also holds
    a second competition with its own first place, a lot nobody bought, and a national
    winner whose auction rounds its score - written with a decimal comma."""
    if year < 2018:
        tables = [
            html_table(["Rank", "Size", "Farm / CWS", "Farmer / Representative", "Region", "Score"],
                       ["1", "27", "Las Fincas Del Suspiro", "A. Zapata", "Coatepec, Veracruz",
                        "90.03"],
                       ["2", "39", "Finca Las Nubes", "L. López", "La Concordia , Chiapas",
                        "88.97"]),
            COE_JUDGES,
            html_table(["Lot #", "Winning Farm / CWS", "Lot Size", "High Bid", "Total Value",
                        "High Bidder(s)"],
                       ["1", "Las Fincas Del Suspiro", "27", "$50.21/lb", "$89,662.01", "Maruyama"],
                       ["2", "Finca Las Nubes", "39", "$15.00/lb", "$38,691.15", "Campos"],
                       ["Totals:", "", "", "", "$128,353.16", ""]),
        ]  # fmt: skip
    else:
        last = year == 2026
        results = ["RANK", "SCORE", "FARM", "FARMER", "REGION", "WEIGHT (kg)", "VARIETY",
                   "PROCESS"]  # fmt: skip
        sales = ["Rank", "Farm", "Score", "Weight (lbs)", "High Bid", "Total Value",
                 "Company Name"]  # fmt: skip
        tables = [
            html_table(results,
                       ["1A", "91.58", "Finca Santa Cruz", "C. Argüello", "La Concordia, Chiapas",
                        "150", "Gesha", "Natural"],
                       ["2", "88,20" if last else "88.20", "Rancho Viejo \u2013 Kohmar", "B. Zilli",
                        "Veracruz", "270", "Typica y Bourbon", "Washed"],
                       *([["3", "87.50", "Unsold", "N. N.", "Puebla", "200", "Marsellesa",
                           "Honey"]] if last else [])),
            html_table(sales,
                       ["1a", "Finca Santa Cruz", "91.58", "330.69", "$92.00", "$30,423.48",
                        "Fisher Coffee"],
                       ["2", "Rancho viejo-Kohmar", "88.2", "595.25", "US$ 14.60", "US$ 8,690.65",
                        "Saza Coffee"]),
            COE_JUDGES,
        ]  # fmt: skip
        if last:
            tables += [
                html_table(results, ["1A", "90.66", "Pocitos", "J. Cadena", "Veracruz", "125",
                                     "Geisha", "Exerimental"]),
                html_table(sales, ["1A", "Pocitos", "90.66", "275.58", "40.7", "11216.11",
                                   "Puente Coffee"]),
                html_table(["Score", "Farm", "Farmer", "Weight (kg)", "Region",
                            "Process, Variety"],
                           ["87.06", "Finca Consuelo", "K. Altamirano", "630", "OCOSINGO",
                            "NATURAL, Borbón"]),
                html_table(["Score", "Farm", "Weight (lbs)", "High Bid", "Total Value",
                            "High Bidder(s)"],
                           ["87.1", "Consuelo", "1,388.91", "$4.00", "$5,555.64", "Nagahama"]),
            ]  # fmt: skip
    page = (
        "<html><body>" + "".join(tables) + "<p>© Alliance. All Rights Reserved.</p></body></html>"
    )
    return page.encode("utf-8")


def enigh_archive(member: str, rows: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, "\r\n".join(rows) + "\r\n")
    return buffer.getvalue()


def census_ageb_archive(rows: list[str] = CENSUS_AGEB_ROWS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CENSUS_AGEB_MEMBER, "﻿" + "\r\n".join(rows) + "\r\n")
    return buffer.getvalue()


# The city's GTFS feed: stops of every system, the Metro's and the Metrobús' named by
# their ids, placed on the boundary fixture's grid of boroughs: Balderas, the Zócalo and
# the Metrobús in Cuauhtémoc's square, Pantitlán in Venustiano Carranza's, Tláhuac in
# its own, La Paz north of them all; a Metrobús station listed once per direction; a
# trolleybus stop the stations leave out.
TRANSIT_STOPS = [
    "stop_id,stop_name,stop_lat,stop_lon,zone_id,wheelchair_boarding",
    "B_0200L1-BALDERAS,Balderas,19.50,-99.25,0200L1-BALDERAS,1",
    "B_0200L1-PANTITLAN,Pantitlán,19.50,-99.00,0200L1-PANTITLAN,1",
    "B_0200L2-ZOCALO,Zócalo,19.51,-99.24,0200L2-ZOCALO,1",
    "B_020L12-TLAHUAC,Tláhuac,19.37,-99.25,020L12-TLAHUAC,1",
    "B_0200LA-LAPAZ,La Paz,19.60,-99.00,0200LA-LAPAZ,1",
    "B_0300L4-20NOVIEMBR,20 de Noviembre,19.5050,-99.2550,0300L4-20NOVIEMBR,1",
    "B_0300L4-20NOVIEMB1,20 de Noviembre,19.5052,-99.2552,0300L4-20NOVIEMB1,1",
    "B_0300L4-PINOSRZSR,Pino Suárez Sur,19.5060,-99.2560,0300L4-PINOSRZSR,1",
    "B_0700T1-CENTRAL,Central de Abasto,19.37,-99.09,0700T1-CENTRAL,1",
]


def gtfs_archive(stops: list[str] = TRANSIT_STOPS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("stops.txt", "\n".join(stops) + "\n")
        archive.writestr("agency.txt", "agency_id,agency_name\nMETRO,Metro\n")  # not read
    return buffer.getvalue()


def broken(text: str) -> str:
    """UTF-8 read as Windows-1252, as the Metro's files of 2021-2023 write it."""
    return text.encode("utf-8").decode("cp1252")


# The Metro's daily entries: a closed station's zero, a line and a station whose name came
# out double-encoded, the Zócalo by its longer name, and a station named twice on one day.
METRO_RIDERSHIP = [
    ("2026-07-30", "Linea 1", "Balderas", 20000),
    ("2026-07-30", "Linea 1", "Pantitlán", 150000),
    ("2026-07-30", "Linea 2", "Zócalo/Tenochtitlan", 30000),
    ("2026-07-30", "Linea 12", "Tláhuac", 0),
    ("2026-07-30", "Linea A", "La Paz", 40000),
    ("2026-07-31", "Linea 1", "Balderas", 22000),
    ("2026-07-31", "Linea 1", "Pantitlán", 158000),
    ("2022-03-01", broken("Línea 1"), broken("Pantitlán"), 90000),
    ("2020-12-15", "Linea 1", "Balderas", 5559),
    ("2020-12-15", "Linea 1", "Balderas", 7529),
]


def metro_csv(rows: list[tuple[str, str, str, int]] = METRO_RIDERSHIP) -> bytes:
    lines = ["fecha,anio,mes,linea,estacion,afluencia"]
    lines += [f"{day},{day[:4]},Julio,{line},{station},{n}" for day, line, station, n in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


# The Metrobús' entries per line: "NaN" before a line opened, a line written two ways.
METROBUS_RIDERSHIP = (
    "fecha,anio,mes,linea,afluencia\n"
    "2005-07-26,2005,Julio,Línea 4,NaN\n"
    "2026-07-30,2026,Julio,Línea 4,80000\n"
    "2026-07-31,2026,Julio,linea 4,82000\n"
).encode()


# FAOSTAT's bulk file of producer prices, as it downloads: one CSV in a ZIP, every field
# quoted. Coffee's year values, a month (left out), an estimated zero (a price nobody
# reported), Mexico's cherry price (SIAP's fixture: 15,000,000 pesos over 5,024 t), a
# country PSD calls otherwise, and another crop priced at exactly the coffee's item code,
# which the reader's quick test lets through and its exact one does not.
FAOSTAT_MEMBER = "Prices_E_All_Data_(Normalized).csv"
FAOSTAT_HEADER = [
    "Area Code", "Area Code (M49)", "Area", "Item Code", "Item Code (CPC)", "Item",
    "Element Code", "Element", "Year Code", "Year", "Months Code", "Months", "Unit",
    "Value", "Flag",
]  # fmt: skip
USD = ("5532", "Producer Price (USD/tonne)", "USD")
LCU = ("5530", "Producer Price (LCU/tonne)", "LCU")
INDEX = ("5539", "Producer Price Index (2014-2016 = 100)", "")


def fao_row(area: tuple[str, str], element: tuple[str, str, str], year: int, value: str,
            flag: str = "A", months: tuple[str, str] = ("7021", "Annual value"),
            item: tuple[str, str] = ("656", "Coffee, green")) -> list[str]:  # fmt: skip
    code, name = area
    element_code, element_name, unit = element
    return [code, f"'{code.zfill(3)}", name, item[0], "'01610", item[1], element_code,
            element_name, str(year), str(year), *months, unit, value, flag]  # fmt: skip


BRAZIL, COLOMBIA, MEXICO_FAO = ("21", "Brazil"), ("44", "Colombia"), ("138", "Mexico")
FAOSTAT_ROWS = [
    fao_row(BRAZIL, USD, 2022, "3163.200000"),
    fao_row(BRAZIL, LCU, 2022, "16300.000000"),
    fao_row(BRAZIL, INDEX, 2022, "0.000000", flag="E"),
    fao_row(BRAZIL, USD, 2022, "3100.000000", months=("7001", "January")),
    fao_row(COLOMBIA, USD, 2023, "3010.800000"),
    fao_row(COLOMBIA, INDEX, 2024, "0.000000", flag="E"),
    fao_row(MEXICO_FAO, LCU, 2024, "2985.700000"),
    fao_row(MEXICO_FAO, USD, 2024, "163.100000"),
    fao_row(("237", "Viet Nam"), LCU, 2023, "15033124.000000"),
    fao_row(("2", "Afghanistan"), LCU, 2023, "656", item=("221", "Almonds, in shell")),
]


def faostat_archive(
    rows: list[list[str]] = FAOSTAT_ROWS, header: list[str] = FAOSTAT_HEADER
) -> bytes:
    text = io.StringIO()
    csv.writer(text, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)
    first = ",".join(header) + "\r\n"  # the header alone is not quoted
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(FAOSTAT_MEMBER, first + text.getvalue())
    return buffer.getvalue()


def siap_year(year: int) -> bytes:
    """An earlier year of SIAP's closing statistics, as it was published: its own headers
    ("Precio" until 2020, "Nomcultivo Sin Um" from 2015 to 2020), Latin-1, CRLF, and one
    coffee row (Ocosingo) beside another crop."""
    crop = "Nomcultivo Sin Um" if 2015 <= year <= 2020 else "Nomcultivo"
    price = "Precio" if year <= 2020 else "Preciomediorural"
    header = (
        "Anio,Idestado,Nomestado,Idddr,Nomddr,Idcader,Nomcader,Idmunicipio,Nommunicipio,"
        "Idciclo,Nomcicloproductivo,Idmodalidad,Nommodalidad,Idunidadmedida,Nomunidad,"
        f"Idcultivo,{crop},Sembrada,Cosechada,Siniestrada,Volumenproduccion,Rendimiento,"
        f"{price},Valorproduccion"
    )
    rows = [
        f"{year},7,Chiapas,23,Palenque,4,Ocosingo,59,Ocosingo,3,Perennes,2,Temporal,200201,"
        f"Tonelada,5710000,Café cereza,2800,2800,0,{3000 + year},1.1,5000,15000000",
        f"{year},7,Chiapas,23,Palenque,4,Ocosingo,59,Ocosingo,1,Primavera-Verano,2,Temporal,"
        "200201,Tonelada,2800000,Maíz grano,100,100,0,200,2,4000,800000",
    ]
    return "\r\n".join([header, *rows, ""]).encode("latin-1")


def zip_fixture(fixture: str, member: str) -> bytes:
    """Rebuild the upstream ZIP envelope around a recorded CSV excerpt."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, (FIXTURES / fixture).read_bytes())
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate_from_the_developers_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """What is installed or configured on this machine must not change what the tests
    exercise: no local MLflow, and no reading the developer's `.env`, which holds real
    credentials and would make a test about a missing one pass or fail by accident."""
    monkeypatch.setenv(
        "MLOPS_MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'unused-mlflow.db').as_posix()}"
    )
    for settings in (Settings, CoffeeCredentials):
        monkeypatch.setitem(settings.model_config, "env_file", None)
    # Which domain runs must come from the test, never from the machine it runs on.
    monkeypatch.delenv("MLOPS_DOMAIN", raising=False)


@pytest.fixture
def coffee_adapter() -> CoffeeAdapter:
    """The real coffee domain, as the core loads it."""
    return domains.coffee.adapter()


@pytest.fixture
def coffee_config(coffee_adapter: CoffeeAdapter) -> CoffeeConfig:
    return coffee_adapter.config


# Every credential the domain can use, so fixtures exercise every source.
FIXTURE_CREDENTIALS = CoffeeCredentials(
    denue_token=SecretStr("fixture-token"), usda_fas_api_key=SecretStr("fixture-key")
)


@pytest.fixture
def recorded() -> dict[str, bytes]:
    """Recorded response body per source name."""
    return {
        "cqi_2018": (FIXTURES / "cqi_2018_sample.csv").read_bytes(),
        "cqi_2023": zip_fixture("cqi_2023_sample.csv", "df_arabica_clean.csv"),
        "psd_coffee": zip_fixture("psd_coffee_sample.csv", "psd_coffee.csv"),
        # Shaped like INEGI's download: a shapefile inside a ZIP, Latin-1 attributes,
        # the layer's own projection. Two boroughs instead of sixteen.
        "cdmx_boroughs": framework_archive(),
        "cdmx_ageb": framework_archive(),  # the same download, another layer of it
        # SIAP's real bytes, still Latin-1: 11 coffee rows (Ocosingo's three CADERs among
        # them) and two other crops, one with nothing harvested.
        "siap_agricola": (FIXTURES / "siap_agricola_sample.csv").read_bytes(),
        "world_bank_prices": xlsx("Monthly Prices", WORLD_BANK_SHEET),
        "ico_prices": pdf(ICO_PAGE),
        "profeco_prices": qqp_archive(QQP_FORTNIGHTS),
        # A closed year's archive is a RAR, which no fixture can be written as: the reader
        # tells an archive by its bytes, so a ZIP stands in for it (the RAR path is
        # tested against bsdtar on its own).
        "profeco_prices_2025": qqp_archive(QQP_CLOSED["2025"]),
        "profeco_prices_2024": qqp_archive(QQP_CLOSED["2024"]),
        # FRED's layout: a day a row, empty where no rate was set. Two of the workbook's
        # months have rates; September has none yet.
        "fred_usd_mxn": b"observation_date,DEXMXUS\n2026-07-01,17.4000\n2026-07-02,17.5000\n"
        b"2026-07-03,\n2026-08-03,17.0000\n2026-08-04,17.1000\n",
        "census_2020": census_archive(),
        "intercensal_2025": survey_archive(),
        "faostat_prices": faostat_archive(),
        "metro_ridership": metro_csv(),
        "metrobus_ridership": METROBUS_RIDERSHIP,
        "transit_stops": gtfs_archive(),
        "census_2020_ageb": census_ageb_archive(),
        "enigh_2024_spending": enigh_archive("gastoshogar.csv", ENIGH_SPENDING),
        "enigh_2024_households": enigh_archive("concentradohogar.csv", ENIGH_HOUSEHOLDS),
        # A page a year; the last one's, as SIAP's (the server serves every year).
        "cup_of_excellence": coe_page(2026),
    }


@pytest.fixture
def server(coffee_config: CoffeeConfig, recorded: dict[str, bytes]) -> RecordedServer:
    urls = {name: str(source.url) for name, source in coffee_config.sources.items()}
    # Behind a redirect, and behind a link.
    elsewhere = {"cqi_2023", "world_bank_prices", "profeco_prices", *PROFECO_CLOSED}
    payloads = {urls[name]: body for name, body in recorded.items() if name not in elsewhere}
    payloads[SIGNED_URL] = recorded["cqi_2023"]
    payloads[urls["world_bank_prices"]] = WORLD_BANK_PAGE
    payloads[WORLD_BANK_FILE] = recorded["world_bank_prices"]
    payloads[urls["profeco_prices"]] = PROFECO_PAGE
    payloads[PROFECO_FILE] = recorded["profeco_prices"]
    for name, url in PROFECO_CLOSED.items():
        payloads[url] = recorded[name]
    # A file a year: the recorded one is the last year's; the earlier ones are made.
    siap = coffee_config.sources["siap_agricola"].editions()
    payloads |= {url: siap_year(year) for year, url, _ in siap[:-1]}
    payloads[siap[-1][1]] = recorded["siap_agricola"]
    competitions = coffee_config.sources["cup_of_excellence"].editions()
    payloads |= {url: coe_page(year) for year, url, _ in competitions}
    # The corpus a publisher serves to anyone. The ones behind a 403 are not here: they
    # are handed over by a person, and their absence is what the extract step reports.
    documents = {
        "pdf": (FIXTURES / "documents" / "sample.pdf").read_bytes(),
        "jats": (FIXTURES / "documents" / "article.xml").read_bytes(),
    }
    payloads |= {
        str(document.url): documents[document.format]
        for document in coffee_config.documents
        if document.inbox is None
    }
    return RecordedServer(
        payloads,
        redirects={urls["cqi_2023"]: SIGNED_URL},
        overpass=(FIXTURES / "overpass_places_sample.json").read_bytes(),
        fas=fas_recording(),
    )


@pytest.fixture
def client(server: RecordedServer) -> Iterator[Any]:
    import httpx

    from mlops_core.data.extract import http_client

    with http_client(httpx.MockTransport(server.handler)) as c:
        yield c


@pytest.fixture
def raw_dir(tmp_path: Path, coffee_config: CoffeeConfig, client: Any) -> Path:
    """A raw layer populated from the recorded payloads, in the real directory layout
    (<data_dir>/<domain>/raw), so `raw_dir.parent` is the domain's data dir.

    The API sources are pulled too, with a token supplied: the clean layer's table of
    places is built from them, so a fixture without them would test half a pipeline. So
    are the documents a publisher serves, exactly as `mlops data extract` pulls them.
    """
    from mlops_core.data.documents import fetch_documents
    from mlops_core.data.extract import extract_all

    data_dir = tmp_path / coffee_config.name
    quick = without_rate_limits(coffee_config)
    extract_all(quick, data_dir / "raw", client)
    adapter = CoffeeAdapter(quick, FIXTURE_CREDENTIALS)
    adapter.extract(data_dir, client)
    fetch_documents(coffee_config.documents, data_dir, client)
    return data_dir / "raw"
