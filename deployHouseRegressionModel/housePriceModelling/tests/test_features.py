from typing import List

import pandas as pd

from houseregression_model.config.core import config
from houseregression_model.pipeline import price_pipe
from houseregression_model.processing.features import SubtractTransformer


def test_temporal_variable_transformer(sample_data: pd.DataFrame) -> None:
    # Given

    transformer = SubtractTransformer(
        variables=config.model_config.temporal_vars,
        target_variable=config.model_config.ref_var,
    )
    assert sample_data["YearRemodAdd"].iat[0] == 1961

    # When

    subject = transformer.fit_transform(sample_data)

    # Then

    assert subject["YearRemodAdd"].iat[0] == 49


def test_pipeline_time_transformer(train_data: List[pd.DataFrame]) -> None:

    # Given
    X_train, X_test, y_train, y_test = train_data

    # When
    transformer = price_pipe.named_steps["elapsed_time"].fit_transform(X_train,
                                                                       y_train)

    # Then
    assert transformer["YearRemodAdd"].iloc[0] == (
        X_train[config.model_config.ref_var].iloc[0] -
        X_train["YearRemodAdd"].iloc[0]
    )
