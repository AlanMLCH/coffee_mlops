"""Shared fixtures. HTTP is replayed from recorded payloads in tests/fixtures/.

`httpx` and the ETL package are imported inside the fixtures that need them, so the
serving tests can run in an environment that installed only the `serving` extra. That
is how CI proves the API's dependency list is complete instead of relying on packages
another pipeline happens to pull in.
"""

import csv
import io
import zipfile
from collections.abc import Iterator
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
        "cdmx_boroughs": (FIXTURES / "cdmx_boroughs_sample.zip").read_bytes(),
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
    extract_all(coffee_config, data_dir / "raw", client)
    adapter = CoffeeAdapter(without_rate_limits(coffee_config), FIXTURE_CREDENTIALS)
    adapter.extract(data_dir, client)
    fetch_documents(coffee_config.documents, data_dir, client)
    return data_dir / "raw"
