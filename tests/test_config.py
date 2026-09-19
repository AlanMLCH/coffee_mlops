from pathlib import Path

import pytest
from pydantic import ValidationError

from coffee_mlops.config import ModelSpec, Settings, load_domain_config


def test_coffee_config_declares_the_stage_1_sources() -> None:
    config = load_domain_config("coffee")

    assert config.name == "coffee"
    assert set(config.sources) == {"cqi_2018", "cqi_2023", "psd_coffee"}


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
