from pathlib import Path

import pytest
from pydantic import ValidationError

from coffee_mlops.config import ModelSpec, Settings, load_domain_config


def test_coffee_config_declares_its_file_sources() -> None:
    config = load_domain_config("coffee")

    assert config.name == "coffee"
    assert set(config.sources) == {"cqi_2018", "cqi_2023", "psd_coffee", "cdmx_boroughs"}
    # The boundary layer is a map, not a table, and says how to read itself.
    boundaries = config.sources["cdmx_boroughs"]
    assert boundaries.spatial is not None
    assert boundaries.spatial.expected_features == 16
    assert config.sources["psd_coffee"].spatial is None


def test_unknown_domain_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError, match="videogames"):
        load_domain_config("videogames")


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "typo.yaml").write_text("name: typo\nsources: {}\ntraget: points\n")

    with pytest.raises(ValidationError, match="traget"):
        load_domain_config("typo", configs_dir=tmp_path)


def test_data_dir_comes_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))

    assert Settings().data_dir == tmp_path


def test_zip_source_must_name_its_member(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nsources:\n  s:\n    url: https://example.com/a.zip\n    filename: a.zip\n"
    )

    with pytest.raises(ValidationError, match="member"):
        load_domain_config("bad", configs_dir=tmp_path)


@pytest.mark.parametrize("leaked", ["aroma", "total_cup_points"])
def test_leaking_columns_cannot_be_declared_as_features(leaked: str) -> None:
    model = load_domain_config("coffee").model.model_dump()
    model["numeric"] = [*model["numeric"], leaked]

    with pytest.raises(ValidationError, match=leaked):
        ModelSpec.model_validate(model)


def test_credentials_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "abc-123")
    monkeypatch.setenv("COFFEE_USDA_FAS_API_KEY", "key-456")

    settings = Settings()

    assert settings.denue_token is not None
    assert settings.denue_token.get_secret_value() == "abc-123"
    assert settings.usda_fas_api_key is not None
    assert settings.usda_fas_api_key.get_secret_value() == "key-456"


def test_a_credential_never_shows_up_by_accident(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DENUE token travels in the URL path, so anything that prints a Settings
    object, logs a traceback or repr's the config must not carry it."""
    monkeypatch.setenv("COFFEE_DENUE_TOKEN", "super-secret-token")

    settings = Settings()

    assert "super-secret-token" not in repr(settings)
    assert "super-secret-token" not in str(settings.denue_token)
    assert "super-secret-token" not in str(settings.model_dump())
