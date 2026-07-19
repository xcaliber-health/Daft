from __future__ import annotations

import json
import pathlib

import pytest

from daft.serve import CatalogSpec, ServeSettings, load_settings


def test_load_settings_from_json(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "host": "0.0.0.0",
        "port": 1234,
        "max_concurrent_queries": 8,
        "catalogs": [{"name": "lake", "properties": {"uri": "http://catalog:8181"}}],
    }
    path = tmp_path / "serve.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("DAFT_SERVE_TOKEN", "sekrit")

    settings = load_settings(path)

    assert settings.host == "0.0.0.0"
    assert settings.port == 1234
    assert settings.max_concurrent_queries == 8
    assert settings.token == "sekrit"
    assert settings.catalogs == (CatalogSpec(name="lake", properties={"uri": "http://catalog:8181"}),)


def test_load_settings_from_yaml(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "serve.yaml"
    path.write_text(
        """
host: 127.0.0.1
port: 9999
disable_plan_payload: true
catalogs:
  - name: lake
    properties:
      uri: http://catalog:8181
      warehouse: s3://bucket/wh
"""
    )
    monkeypatch.delenv("DAFT_SERVE_TOKEN", raising=False)

    settings = load_settings(path)

    assert settings.port == 9999
    assert settings.token is None
    assert settings.disable_plan_payload is True
    assert settings.catalogs[0].properties["warehouse"] == "s3://bucket/wh"


def test_load_settings_defaults_when_fields_missing(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "serve.json"
    path.write_text("{}")
    monkeypatch.delenv("DAFT_SERVE_TOKEN", raising=False)

    settings = load_settings(path)

    assert settings == ServeSettings()


@pytest.mark.parametrize(
    "config",
    [
        '{"catalogs": "nope"}',
        '{"catalogs": [{"properties": {}}]}',
        '{"catalogs": [{"name": "x", "properties": "nope"}]}',
        '{"port": "not-a-number"}',
        '{"host": 42}',
        '["not", "a", "mapping"]',
    ],
)
def test_load_settings_rejects_invalid_configuration(
    tmp_path: pathlib.Path, config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "serve.json"
    path.write_text(config)
    monkeypatch.delenv("DAFT_SERVE_TOKEN", raising=False)

    with pytest.raises((ValueError, TypeError)):
        load_settings(path)


def test_token_env_variable_name_is_configurable(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "serve.json"
    path.write_text("{}")
    monkeypatch.setenv("MY_TOKEN", "abc")
    monkeypatch.delenv("DAFT_SERVE_TOKEN", raising=False)

    assert load_settings(path, token_env="MY_TOKEN").token == "abc"
