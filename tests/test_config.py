from pathlib import Path

import pytest
from pydantic import ValidationError

import domains.coffee
from domains.coffee.config import CleaningConfig, CoffeeConfig, CoffeeCredentials, ShopConfig
from mlops_core.adapter import available_domains, load_adapter
from mlops_core.config import ModelSpec, Settings, load_config


def test_coffee_config_declares_its_file_sources() -> None:
    config = load_adapter("coffee").config

    assert config.name == "coffee"
    assert set(config.sources) == {
        "cqi_2018",
        "cqi_2023",
        "psd_coffee",
        "cdmx_boroughs",
        "siap_agricola",
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


def test_a_lone_domain_is_used_when_none_is_named() -> None:
    assert available_domains() == ["coffee"]
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
