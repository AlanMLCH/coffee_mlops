from pathlib import Path

import pytest
from pydantic import ValidationError

import domains.coffee
from domains.coffee.config import CleaningConfig, CoffeeConfig, CoffeeCredentials, ShopConfig
from mlops_core.adapter import available_domains, load_adapter
from mlops_core.config import (
    ModelSpec,
    Settings,
    env_file_names,
    load_config,
    unread_settings,
)


def test_coffee_config_declares_its_file_sources() -> None:
    config = load_adapter("coffee").config

    assert config.name == "coffee"
    assert set(config.sources) == {
        "cqi_2018",
        "cqi_2023",
        "psd_coffee",
        "cdmx_boroughs",
        "siap_agricola",
        "world_bank_prices",
        "ico_prices",
        "profeco_prices",
        "profeco_prices_2024",
        "profeco_prices_2025",
        "fred_usd_mxn",
        "census_2020",
        "intercensal_2025",
        "faostat_prices",
        "metro_ridership",
        "metrobus_ridership",
        "transit_stops",
        "census_2020_ageb",
        "cdmx_ageb",
        "enigh_2024_spending",
        "enigh_2024_households",
        "cup_of_excellence",
    }
    # The boundary layer is a map, not a table, and says how to read itself.
    boundaries = config.sources["cdmx_boroughs"]
    assert boundaries.spatial is not None
    assert boundaries.spatial.expected_features == 16
    assert config.sources["psd_coffee"].spatial is None


def test_each_model_names_its_own_tables() -> None:
    review = load_adapter("coffee").config.model_named("review")

    assert (review.features_table, review.predictions_table) == (
        "review_features",
        "review_predictions",
    )
    assert review.keys == ["review_id", "snapshot", "grading_date"]


def test_a_lone_domain_is_used_when_none_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    assert available_domains() == ["coffee", "coffee_shop"]
    monkeypatch.setattr("mlops_core.adapter.available_domains", lambda: ["coffee"])

    assert load_adapter().config.name == "coffee"


def test_unknown_domain_fails_loudly() -> None:
    with pytest.raises(ValueError, match="No domain 'videogames'"):
        load_adapter("videogames")


def test_a_domain_whose_own_import_breaks_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo inside a domain must not read as "that domain does not exist"."""

    def broken(name: str) -> None:
        raise ModuleNotFoundError("No module named 'shapely'", name="shapely")

    monkeypatch.setattr("mlops_core.adapter.importlib.import_module", broken)

    with pytest.raises(ModuleNotFoundError, match="shapely"):
        load_adapter("coffee")


def test_a_missing_config_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nowhere.yaml"):
        load_config(tmp_path / "nowhere.yaml", CoffeeConfig)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    """A domain's own sections are validated as strictly as the core's."""
    yaml = domains.coffee.CONFIG_PATH.read_text(encoding="utf-8") + "\ntraget: points\n"
    (tmp_path / "typo.yaml").write_text(yaml, encoding="utf-8")

    with pytest.raises(ValidationError, match="traget"):
        load_config(tmp_path / "typo.yaml", CoffeeConfig)


def test_data_dir_comes_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLOPS_DATA_DIR", str(tmp_path))

    assert Settings().data_dir == tmp_path


def test_zip_source_must_name_its_member(tmp_path: Path) -> None:
    yaml = domains.coffee.CONFIG_PATH.read_text(encoding="utf-8").replace(
        "    member: psd_coffee.csv\n", "", 1
    )
    (tmp_path / "bad.yaml").write_text(yaml, encoding="utf-8")

    with pytest.raises(ValidationError, match="member"):
        load_config(tmp_path / "bad.yaml", CoffeeConfig)


def test_a_squarespace_shop_must_say_where_its_store_is() -> None:
    """Squarespace has no fixed catalog path, unlike Shopify's `/products.json`."""
    with pytest.raises(ValidationError, match="store_path"):
        ShopConfig(shop="nowhere", platform="squarespace", base_url="https://shop.test")


def test_a_document_filed_under_an_unknown_topic_is_refused() -> None:
    """A topic nobody declared would route nothing and be found by nobody."""
    config = load_adapter("coffee").config.model_dump()
    config["documents"][0]["topics"] = ["barista history"]

    with pytest.raises(ValidationError, match="barista history"):
        CoffeeConfig.model_validate(config)


def test_two_documents_cannot_share_a_name() -> None:
    """They would share a raw folder, and each ingestion would look like a change."""
    config = load_adapter("coffee").config.model_dump()
    config["documents"].append(config["documents"][0])

    with pytest.raises(ValidationError, match="Document names must be unique"):
        CoffeeConfig.model_validate(config)


def test_the_roasters_processes_speak_the_cqi_vocabulary() -> None:
    """Otherwise a roaster's "washed" and a graded lot's could not be compared."""
    cleaning = load_adapter("coffee").config.cleaning.model_dump()
    cleaning["roaster_sheets"]["processes"]["fermented"] = "ferment"

    with pytest.raises(ValidationError, match="fermented"):
        CleaningConfig.model_validate(cleaning)


@pytest.mark.parametrize("leaked", ["aroma", "total_cup_points"])
def test_leaking_columns_cannot_be_declared_as_features(leaked: str) -> None:
    model = load_adapter("coffee").config.model_named("review").spec.model_dump()
    model["numeric"] = [*model["numeric"], leaked]

    with pytest.raises(ValidationError, match=leaked):
        ModelSpec.model_validate(model)


def test_credentials_belong_to_the_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the domain's prefix, not the core's: another domain brings its own keys."""
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "abc-123")
    monkeypatch.setenv("COFFEE_USDA_FAS_API_KEY", "key-456")

    keys = CoffeeCredentials()

    assert keys.denue_token is not None
    assert keys.denue_token.get_secret_value() == "abc-123"
    assert keys.usda_fas_api_key is not None
    assert keys.usda_fas_api_key.get_secret_value() == "key-456"
    assert not hasattr(Settings(), "denue_token")


def test_a_credential_never_shows_up_by_accident(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DENUE token travels in the URL path, so anything that prints the credentials,
    logs a traceback or repr's them must not carry it."""
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "super-secret-token")

    keys = CoffeeCredentials()

    assert "super-secret-token" not in repr(keys)
    assert "super-secret-token" not in str(keys.denue_token)
    assert "super-secret-token" not in str(keys.model_dump())


def test_with_two_domains_installed_a_command_must_say_which(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picking the lone domain is a convenience; with two it would be a guess."""
    monkeypatch.setattr("mlops_core.adapter.available_domains", lambda: ["coffee", "tea"])

    with pytest.raises(ValueError, match="Name a domain"):
        load_adapter()


def test_a_bags_shop_is_described_with_every_shop_the_config_reads() -> None:
    """The agent's model picks the shop's word from this description: a shop added to the
    config and missing here would be priced as a shop the model never saw."""
    from domains.coffee.request import Offer

    config = load_adapter("coffee").config
    described = Offer.model_fields["shop"].description or ""

    assert config.roasters is not None
    assert all(shop.shop in described for shop in config.roasters.shops)


def test_a_models_example_is_a_request_body_as_json_carries_it() -> None:
    """YAML reads 2026-09-01 as a date; a request body has no dates, only text."""
    example = load_adapter("coffee").config.model_named("green_price").example

    assert example == {"indicator": "other_milds", "month": "2026-09-01"}


def test_a_workbook_names_its_sheet_and_a_link_is_a_pattern() -> None:
    from mlops_core.config import SourceConfig

    SourceConfig(url="https://b.test/p.xlsx", filename="p.xlsx", sheet="Prices")
    with pytest.raises(ValidationError, match="a workbook names its `sheet`"):
        SourceConfig(url="https://b.test/p.xlsx", filename="p.xlsx")
    with pytest.raises(ValidationError, match="a workbook names its `sheet`"):
        SourceConfig(url="https://b.test/p.csv", filename="p.csv", sheet="Prices")
    with pytest.raises(ValidationError, match="unterminated"):
        SourceConfig(url="https://b.test/", filename="p.csv", link=r"prices-(\d+")


def test_a_file_a_year_writes_its_year_where_it_changes() -> None:
    from mlops_core.config import SourceConfig

    yearly = {"url": "https://s.test/?ANIO={year}", "filename": "cierre_{year}.csv"}
    source = SourceConfig(**yearly, years={"first": 2023, "last": 2025})

    assert [(y, f) for y, _, f in source.editions()] == [
        (2023, "cierre_2023.csv"), (2024, "cierre_2024.csv"), (2025, "cierre_2025.csv")
    ]  # fmt: skip
    assert source.editions()[0][1] == "https://s.test/?ANIO=2023"
    assert SourceConfig(url="https://s.test/", filename="one.csv").editions() == []
    with pytest.raises(ValidationError, match="go together"):
        SourceConfig(**yearly)  # a year in the address, and no years
    with pytest.raises(ValidationError, match="go together"):
        SourceConfig(url="https://s.test/", filename="one.csv", years={"first": 1, "last": 2})
    with pytest.raises(ValidationError, match="not found by a link"):
        SourceConfig(**yearly, years={"first": 2023, "last": 2025}, link="x")
    with pytest.raises(ValidationError, match="2025 > 2023"):
        SourceConfig(**yearly, years={"first": 2025, "last": 2023})
    # In a path, the year's braces are escaped by the URL type, and still the year.
    paged = SourceConfig(url="https://c.test/mexico-{year}/", filename="mexico-{year}.html",
                         years={"first": 2012, "last": 2013})  # fmt: skip
    assert [u for _, u, _ in paged.editions()] == ["https://c.test/mexico-2012/",
                                                   "https://c.test/mexico-2013/"]  # fmt: skip
    # A year never published is skipped, not a download that fails; the last one cannot be.
    held = SourceConfig(**yearly, years={"first": 2023, "last": 2025, "missing": [2024]})
    assert [y for y, _, _ in held.editions()] == [2023, 2025]
    with pytest.raises(ValidationError, match=r"missing years \[2025\]"):
        SourceConfig(**yearly, years={"first": 2023, "last": 2025, "missing": [2025]})
    with pytest.raises(ValidationError, match="thousands"):
        SourceConfig(url="https://s.test/", filename="one.csv", thousands=",,")


def test_a_link_is_found_by_where_it_points_or_by_what_it_says_not_both() -> None:
    from mlops_core.config import SourceConfig

    page = {"url": "https://b.test/", "filename": "p.zip", "member": "p"}
    SourceConfig(**page, link_text=r"^Precios 2026$")
    with pytest.raises(ValidationError, match="not both"):
        SourceConfig(**page, link=r"\.zip$", link_text=r"^Precios 2026$")
    with pytest.raises(ValidationError, match="`link_text` is not a pattern"):
        SourceConfig(**page, link_text=r"Precios (2026")


def test_shelf_prices_come_from_a_source_that_downloads_their_folder() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["consumer_prices"]["source"] = "ico_prices"  # a PDF, not an archive of tables

    with pytest.raises(ValidationError, match="names the archive's folder of fortnights"):
        CoffeeConfig.model_validate(config)


def test_producer_prices_come_from_a_source_that_names_the_csv_in_its_zip() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["producer_prices"]["source"] = "fred_usd_mxn"  # a CSV on its own

    with pytest.raises(ValidationError, match="names the CSV inside FAOSTAT's ZIP"):
        CoffeeConfig.model_validate(config)


def test_the_borough_profile_reads_a_source_that_exists() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["borough_profile"]["source"] = "intercensal_2030"

    with pytest.raises(ValidationError, match="reads 'intercensal_2030', which is not a source"):
        CoffeeConfig.model_validate(config)


def test_transit_reads_sources_that_exist() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["transit"]["metrobus"] = "trolleybus_ridership"

    with pytest.raises(ValidationError, match=r"`transit` reads \['trolleybus_ridership'\]"):
        CoffeeConfig.model_validate(config)


def test_the_zones_read_a_census_in_a_zip_and_draw_a_map_layer() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["census_zones"]["census"] = "fred_usd_mxn"  # a CSV on its own

    with pytest.raises(ValidationError, match="names the CSV inside INEGI's ZIP"):
        CoffeeConfig.model_validate(config)

    config["census_zones"] |= {"census": "census_2020_ageb", "layer": "census_2020"}
    with pytest.raises(ValidationError, match="it has to be a map layer"):
        CoffeeConfig.model_validate(config)


def test_two_survey_columns_cannot_share_a_name() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["borough_profile"]["indicators"]["PCN_VPH_ALQUI"]["name"] = "owned_pct"

    with pytest.raises(ValidationError, match=r"names two columns the same: \['owned_pct'\]"):
        CoffeeConfig.model_validate(config)


def test_what_a_presentation_says_is_read_with_patterns_that_compile() -> None:
    config = load_adapter("coffee").config.model_dump()
    config["consumer_prices"]["decaf"] = "descafeinad(o"

    with pytest.raises(ValidationError, match=r"`consumer_prices\.decaf` is not a pattern"):
        CoffeeConfig.model_validate(config)


def test_a_schedule_is_a_cron_in_a_timezone() -> None:
    from mlops_core.config import ScheduleConfig

    schedule = load_adapter("coffee").config.schedule
    assert schedule == ScheduleConfig(data="0 7 * * *", timezone="America/Mexico_City")
    with pytest.raises(ValidationError, match="should match pattern"):
        ScheduleConfig(data="every morning", timezone="UTC")


def test_a_setting_under_the_wrong_prefix_or_name_is_said_to_be_unread(tmp_path: Path) -> None:
    """Pydantic ignores them in silence; the default then passes for the value."""
    env = tmp_path / ".env"
    env.write_text(
        "# a comment = not a variable\nCOFFEE_DATA_DIR=data\nMLOPS_DATADIR=x\n"
        "COFFEE_DENUE_TOKEN=secret\nMLOPS_DATA_DIR=data\n",
        encoding="utf-8",
    )

    names = env_file_names(env)

    assert names == ["COFFEE_DATA_DIR", "MLOPS_DATADIR", "COFFEE_DENUE_TOKEN", "MLOPS_DATA_DIR"]
    assert unread_settings(names, "coffee") == {
        "COFFEE_DATA_DIR": "not read: the setting is MLOPS_DATA_DIR",
        "MLOPS_DATADIR": "no setting has this name",
    }
    assert env_file_names(tmp_path / "missing.env") == []
