"""INEGI's 2020 Census by urban AGEB: the city in pieces of a few dozen blocks.

INEGI publishes the census' principal results per urban AGEB and per block in one CSV for
Mexico City (`conjunto_de_datos_ageb_urbana_09_cpv2020.csv` inside a 13 MB ZIP): 68,941
rows x 230 columns, UTF-8 with a mark, verified 2026-09-30. A row per block, a total row
per AGEB ("Total AGEB urbana", 2,433 of them), and totals per borough, locality and the
state. Only the AGEB totals and the columns the zones use are kept: the whole file as text
would hold some 16 million cells to keep a fraction of them, and the clean layer already
runs close to the laptop's memory. A figure INEGI withholds to protect a few households
is written "*", read as null.
"""

import zipfile
from pathlib import Path

import polars as pl

AGEB_TOTAL = "Total AGEB urbana"
# The key, and the figures `census_zones` keeps, in INEGI's names.
COLUMNS = ["ENTIDAD", "MUN", "LOC", "AGEB", "NOM_LOC", "POBTOT", "TVIVHAB", "GRAPROES", "PEA",
           "POB65_MAS", "VPH_INTER", "VPH_AUTOM", "VPH_PC"]  # fmt: skip


def read_census_zones(path: Path, member: str) -> pl.DataFrame:
    """Each urban AGEB's total row of the census, as text, withheld figures as nulls."""
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    frame = pl.read_csv(data, infer_schema_length=0, null_values=["*"], n_rows=0)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{member} has no {missing}: INEGI changed its layout")
    frame = pl.read_csv(data, columns=COLUMNS, infer_schema_length=0, null_values=["*"])
    return frame.filter(pl.col("NOM_LOC") == AGEB_TOTAL)
