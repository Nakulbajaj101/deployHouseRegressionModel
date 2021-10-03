import pytest

from houseregression_model.config.core import config
from houseregression_model.processing.utility_functions import load_dataset


@pytest.fixture()
def sample_data():
    data = load_dataset(filename=config.app_config.test_data_file)
    return data
