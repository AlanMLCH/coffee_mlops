from pathlib import Path

import pytest
from pydantic import ValidationError

from coffee_mlops.config import Settings, load_domain_config


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
