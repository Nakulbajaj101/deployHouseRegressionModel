import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from houseregression_model.config.core import config
from houseregression_model.processing.utility_functions import load_dataset


@pytest.fixture()
def sample_data():
    data = load_dataset(filename=config.app_config.test_data_file)
    return data


@pytest.fixture()
def sample_validation_data():
    data = load_dataset(filename=config.app_config.training_data_file)
    validation_data = data.copy()[-200:]
    return validation_data


@pytest.fixture()
def old_predictions():
    data = pd.read_csv("houseregression_model/old_predictions.csv", sep=",")
    return data


@pytest.fixture()
def train_data():
    data = load_dataset(filename=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    return X_train, X_test, y_train, y_test
