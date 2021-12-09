import logging
import math

import numpy as np
import pandas as pd

from houseregression_model.predict import make_predictions

logging.basicConfig(level=logging.INFO)


def test_make_prediction(sample_data: pd.DataFrame) -> None:
    """Test for testing predictions and expected values"""

    # Given
    expected_first_prediction_value = 125757
    expected_no_predictions = sample_data.shape[0]

    # When
    result = make_predictions(input_data=sample_data)

    # Then
    predictions = result.get("predictions")

    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert math.isclose(len(predictions), expected_no_predictions, abs_tol=20)
    assert math.isclose(predictions[0],
                        expected_first_prediction_value,
                        abs_tol=100)
