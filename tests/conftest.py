from __future__ import annotations

import pytest
from langchain_tests.conftest import (
    CustomPersister,
    CustomSerializer,
    base_vcr_config,
)
from vcr import VCR


def _remove_response_headers(response: dict) -> dict:
    response["headers"] = {}
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    config = base_vcr_config()
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    config["before_record_response"] = _remove_response_headers
    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    del config
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
