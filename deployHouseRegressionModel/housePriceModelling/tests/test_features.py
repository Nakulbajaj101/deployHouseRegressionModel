import pandas as pd

from houseregression_model.config.core import config
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
