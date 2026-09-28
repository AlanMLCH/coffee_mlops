"""Tasting notes read from the roasters' descriptions into the SCA's flavour categories.

The rules are small and written by hand, so each gets a sentence that would fool a
looser reader: a producer's story, a surname, a negation, the cherry that is picked.
"""

import re

import polars as pl
import pytest
from pydantic import ValidationError

from domains.coffee.config import CoffeeConfig, FlavorGroup, TastingNotesConfig
from domains.coffee.flavors import folded, inflected, tasting_notes
from domains.coffee.schemas import clean_schemas
from mlops_core.adapter import load_adapter
from mlops_core.contracts import check_contract

RULES = TastingNotesConfig(
    cues=["notas", "sabe", "aroma", "en taza"],
    negations=["sin", "ni"],
    title_notes=["cucurucho"],
    other_meanings=["cerezas? maduras", "de (?:la|las) cerezas?"],
    groups=[
        FlavorGroup(category="fruity", subcategory="berry", notes={"fruto rojo": "red berries"}),
        FlavorGroup(category="fruity", notes={"cereza": "cherry", "durazno": "peach"}),
        FlavorGroup(category="floral", notes={"jamaica": "hibiscus", "hibisco": "hibiscus"}),
        FlavorGroup(category="sweet", notes={"caramelizado": "caramelised", "caramelo": "caramel"}),
        FlavorGroup(category="nutty_cocoa", notes={"nuez": "walnut", "chocolate": "chocolate"}),
        FlavorGroup(category="spice", notes={"nuez moscada": "nutmeg"}),
        FlavorGroup(category="sour_fermented", notes={"fermentado": "fermented"}),
        FlavorGroup(category="other", notes={"tierra mojada": "wet earth", "flor": "flowers"}),
    ],
)


def notes(description: str, title: str = "Un café", shop: str = "almanegra") -> list[str]:
    coffees = pl.DataFrame(
        {"coffee_id": ["c"], "shop": [shop], "title": [title], "description": [description]}
    )
    return tasting_notes(coffees, RULES)["note_en"].to_list()


def test_a_note_counts_only_after_a_cue_and_within_its_sentence() -> None:
    """A producer's story names fruit and land without tasting of them."""
    story = "La familia cosecha durazno y chocolate en su tierra mojada. "
    assert notes(story + "Un café con notas a caramelo. Nuez para la cena.") == ["caramel"]


def test_plurals_and_the_other_gender_are_the_same_note() -> None:
    found = notes("Notas a frutos rojos y a nueces caramelizadas.")
    assert sorted(found) == ["caramelised", "red berries"]  # "nueces" is not "nuez" + s


def test_the_longest_note_wins() -> None:
    assert notes("Sabe a nuez moscada.") == ["nutmeg"]


def test_a_negated_cue_opens_nothing_and_a_negation_ends_a_list() -> None:
    assert notes("Limpio, ni llegar a los sabores sin notas a fermentado.") == []
    assert notes("Notas a durazno, sin nada fermentado.") == ["peach"]


def test_a_capitalised_word_mid_sentence_is_a_name() -> None:
    """Juan Carlos Flores is not floral; a capitalised first item of a list is a note."""
    assert notes("Notas a chocolate del productor Juan Carlos Flor.") == ["chocolate"]
    assert sorted(notes("En taza: Chocolate, Caramelo")) == ["caramel", "chocolate"]


def test_the_cherry_that_is_picked_is_not_a_note() -> None:
    found = notes("Aroma de las cerezas maduras. Notas de cereza y durazno.")
    assert sorted(found) == ["cherry", "peach"]
    assert notes("Aroma de las cerezas.") == []


def test_only_some_shops_write_their_notes_in_the_title() -> None:
    title = "Chiapas- Caramelo y durazno"
    assert sorted(notes("", title, shop="cucurucho")) == ["caramel", "peach"]
    assert notes("", title) == []  # the same words from another shop are a name


def test_a_note_is_kept_once_whatever_its_spelling_and_where_it_was_first_written() -> None:
    coffees = pl.DataFrame(
        {
            "coffee_id": ["c"],
            "shop": ["cucurucho"],
            "title": ["Oaxaca- Jamaica"],
            "description": ["Notas a hibisco y a jamaica."],
        }
    )

    table = tasting_notes(coffees, RULES)

    assert table.select("note", "note_en", "category", "source").rows() == [
        ("jamaica", "hibiscus", "floral", "title")
    ]


def test_coffees_without_notes_leave_an_empty_table_with_its_columns() -> None:
    table = tasting_notes(pl.DataFrame(schema=["coffee_id", "shop", "title", "description"]), RULES)
    assert table.is_empty() and table.columns[:3] == ["coffee_id", "shop", "note"]
    assert notes(None) == []  # type: ignore[arg-type]  # a coffee with no description


def test_inflection_and_folding() -> None:
    assert re.fullmatch(inflected("fruto rojo"), "frutos rojos")
    assert re.fullmatch(inflected("fruta con hueso"), "frutas con hueso")
    assert re.fullmatch(inflected("caramelizado"), "caramelizadas")
    assert re.fullmatch(inflected("limón"), "limones")
    assert re.fullmatch(inflected("té negro"), "te negro")
    assert not re.fullmatch(inflected("fruta con hueso"), "frutas cones hueso")
    # One character for each: positions in the folded text are positions in the original.
    assert folded("Café ﬀ Ñ") == "cafe f n"


def test_the_real_lexicon_reads_the_real_contract(coffee_config: CoffeeConfig) -> None:
    rules = coffee_config.cleaning.tasting_notes
    coffees = pl.DataFrame(
        {
            "coffee_id": ["a", "b"],
            "shop": ["almanegra", "cucurucho"],
            "title": ["Kenia Kiambu", "Chiapas- Caramelo, avellana y chocolate"],
            "description": [
                "Un café con notas a frambuesa, té negro y azúcar morena. "
                "Cosecha de cerezas maduras.",
                None,
            ],
        }
    )

    table = check_contract(
        clean_schemas(coffee_config.cleaning)["roaster_flavors"], tasting_notes(coffees, rules)
    )

    by_coffee = table.group_by("coffee_id").agg(pl.col("note_en").sort()).sort("coffee_id")
    assert by_coffee["note_en"].to_list() == [
        ["black tea", "brown sugar", "raspberry"],
        ["caramel", "chocolate", "hazelnut"],
    ]
    assert table.filter(pl.col("note_en") == "raspberry")["subcategory"].item() == "berry"


def test_a_note_cannot_be_in_two_groups() -> None:
    groups = [*RULES.groups, FlavorGroup(category="sweet", notes={"Durazno": "peach"})]
    with pytest.raises(ValidationError, match=r"'Durazno' is in fruity and sweet"):
        TastingNotesConfig.model_validate(RULES.model_dump() | {"groups": groups})


def test_another_meaning_has_to_be_a_pattern() -> None:
    with pytest.raises(ValidationError, match="not a pattern"):
        TastingNotesConfig.model_validate(RULES.model_dump() | {"other_meanings": ["(cereza"]})


def test_the_shops_that_title_their_notes_have_to_exist() -> None:
    """A misspelt shop would have its titles never read, and nothing would say so."""
    config = load_adapter("coffee").config.model_dump()
    config["cleaning"]["tasting_notes"]["title_notes"] = ["cucurucho", "cucuruch0"]

    with pytest.raises(ValidationError, match="cucuruch0"):
        CoffeeConfig.model_validate(config)


def test_categories_keep_the_order_the_config_gives() -> None:
    assert RULES.categories[:3] == ["fruity", "floral", "sweet"]
