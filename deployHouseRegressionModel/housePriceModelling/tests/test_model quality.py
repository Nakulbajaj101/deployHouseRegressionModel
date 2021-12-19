import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from houseregression_model.config.core import config
from houseregression_model.predict import make_predictions

logging.basicConfig(level=logging.INFO)


def test_prediction_quality(
    sample_validation_data: pd.DataFrame, old_predictions: pd.DataFrame
) -> None:

    # Givem
    old_predictions = [val for val in old_predictions["old_model_predictions"]]

    # When
    new_predictions_results = make_predictions(input_data=sample_validation_data)

    new_predictions = new_predictions_results.get("predictions")

    true_values = sample_validation_data[config.model_config.target]

    old_mse = mean_squared_error(true_values, old_predictions, squared=False)
    new_mse = mean_squared_error(true_values, new_predictions, squared=False)

    old_r2 = r2_score(true_values, old_predictions)
    new_r2 = r2_score(true_values, new_predictions)

    # Then

    # Make sure new predictions are better
    assert new_mse < old_mse
    assert new_r2 > old_r2

    # The Model shouldnt deviate too much
    assert math.isclose(np.std(old_predictions), np.std(new_predictions), abs_tol=2000)

    assert math.isclose(
        np.mean(old_predictions), np.mean(new_predictions), abs_tol=3000
    )

    # The R2 should and mse should meet certain threshold
    assert new_r2 > 0.75
    assert new_mse < 10000


def test_model_benchmarking(sample_validation_data: pd.DataFrame) -> None:

    # Given
    predictions_results = make_predictions(input_data=sample_validation_data)

    predictions = predictions_results.get("predictions")

    true_values = sample_validation_data[config.model_config.target]

    small_difference = 10000
    medium_difference = 50000
    large_difference = 100000

    # When
    value_difference = [
        abs(true - pred) for true, pred in zip(true_values, predictions)
    ]

    small_difference_total = [val for val in value_difference if val > small_difference]

    medium_difference_total = [
        val for val in value_difference if val > medium_difference
    ]

    large_difference_total = [val for val in value_difference if val > large_difference]

    data_length = sample_validation_data.shape[0]

    # Then

    # Only few predictions should be allowed to deviate by too much
    assert len(small_difference_total) < int(0.2 * data_length)
    assert len(medium_difference_total) < int(0.02 * data_length)
    assert len(large_difference_total) == 0
