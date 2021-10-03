import pandas as pd
import pytest
from app.main import app
from fastapi.testclient import TestClient
from houseregression_model.config.core import config
from houseregression_model.processing.utility_functions import load_dataset


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame():
    return load_dataset(filename=config.app_config.test_data_file)


@pytest.fixture()
def client() -> TestClient:
    _client = TestClient(app)
    return _client
