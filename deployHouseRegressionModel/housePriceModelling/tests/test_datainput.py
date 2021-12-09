from houseregression_model.processing.validation import validate_inputs


def test_validate_inputs(sample_data):
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_data)

    # Then
    assert not errors

    # we expect that 2 rows are removed due to missing vars
    # 1459 is the total number of rows in the test data set (test.csv)
    # and 1457 number returned after 2 rows are filtered out.
    assert len(sample_data) == 1459
    assert len(validated_inputs) == 1449


def test_validate_inputs_identifies_errors(sample_data):
    # Given
    test_inputs = sample_data.copy()

    # introduce errors
    test_inputs.at[0, "YearRemodAdd"] = "1871"  # we expect a string

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors is None
