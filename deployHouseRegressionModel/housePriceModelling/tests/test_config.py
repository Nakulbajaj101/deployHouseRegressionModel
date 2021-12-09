#
import logging
from pathlib import Path

import pytest
from houseregression_model.config.core import (create_and_validate_config,
                                               fetch_config_from_yaml)
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO)


TEST_APP_CONFIG = """
# Package Overview
package_name: regression_model

"""

TEST_MODEL_CONFIG = """

# Package Overview
package_name: regression_model
training_data_file: train.csv
test_data_file: test.csv
pipeline_save_file: regression_model_output_v

# Target Param
target: SalePrice
"""


INVALID_TEST_CONFIG = """
# Package Overview
package_name: regression_model

# Data Files
training_data_file: train.csv
test_data_file: test.csv

# Variables
# The variable we are attempting to predict (sale price)
target: SalePrice

pipeline_name: regression_model
pipeline_save_file: regression_model_output_v

# Will cause syntax errors since they begin with numbers
variables_to_rename:
  1stFlrSF: FirstFlrSF
  2ndFlrSF: SecondFlrSF
  3SsnPorch: ThreeSsnPortch

features:
  - MSSubClass
  - MSZoning
  - LotFrontage
  - LotShape
  - LandContour
  - LotConfig
  - Neighborhood
  - OverallQual
  - OverallCond
  - YearRemodAdd
  - RoofStyle
  - Exterior1st
  - ExterQual
  - Foundation
  - BsmtQual
  - BsmtExposure
  - BsmtFinType1
  - HeatingQC
  - CentralAir
  - FirstFlrSF  # renamed
  - SecondFlrSF  # renamed
  - GrLivArea
  - BsmtFullBath
  - HalfBath
  - KitchenQual
  - TotRmsAbvGrd
  - Functional
  - Fireplaces
  - FireplaceQu
  - GarageFinish
  - GarageCars
  - GarageArea
  - PavedDrive
  - WoodDeckSF
  - ScreenPorch
  - SaleCondition
  # this one is only to calculate temporal variable:
  - YrSold

# set train/test split
test_size: 0.1

# to set the random seed
random_state: 0

alpha: 0.001
loss: ls
learning_rate: 0.1
n_estimators: 500

# allowed loss functions
allowed_loss_functions:
  - huber

# categorical variables with NA in train set
categorical_vars_with_na_frequent:
  - BsmtQual
  - BsmtExposure
  - BsmtFinType1
  - GarageFinish

categorical_vars_with_na_missing:
  - FireplaceQu

numerical_vars_with_na:
  - LotFrontage

temporal_vars:
  - YearRemodAdd

ref_var: YrSold


# variables to log transform
numericals_log_vars:
  - LotFrontage
  - FirstFlrSF
  - GrLivArea

binarize_vars:
  - ScreenPorch

# variables to map
qual_vars:
  - ExterQual
  - BsmtQual
  - HeatingQC
  - KitchenQual
  - FireplaceQu

exposure_vars:
  - BsmtExposure

finish_vars:
  - BsmtFinType1

garage_vars:
  - GarageFinish

categorical_vars:
  - MSSubClass
  - MSZoning
  - LotShape
  - LandContour
  - LotConfig
  - Neighborhood
  - RoofStyle
  - Exterior1st
  - Foundation
  - CentralAir
  - Functional
  - PavedDrive
  - SaleCondition

# variable mappings
qual_mappings:
  Po: 1
  Fa: 2
  TA: 3
  Gd: 4
  Ex: 5
  Missing: 0
  NA: 0

exposure_mappings:
  No: 1
  Mn: 2
  Av: 3
  Gd: 4


finish_mappings:
  Missing: 0
  NA: 0
  Unf: 1
  LwQ: 2
  Rec: 3
  BLQ: 4
  ALQ: 5
  GLQ: 6


garage_mappings:
  Missing: 0
  NA: 0
  Unf: 1
  RFn: 2
  Fin: 3
"""


def test_read_config():

    # Given
    parsed_config = fetch_config_from_yaml()

    # When
    config = create_and_validate_config(parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_validation_error(tmpdir):

    # Given
    configs_dir = Path(tmpdir)
    config = configs_dir / "sample_config.yml"
    config.write_text(INVALID_TEST_CONFIG)

    parsed_config = fetch_config_from_yaml(config)

    # When
    with pytest.raises(ValidationError) as e:
        create_and_validate_config(parsed_config)

    # Then
    assert "is not in the allowed set" in str(e.value)


def test_missing_config_app_fields_raises_error(tmpdir):

    # Given
    configs_dir = Path(tmpdir)
    config = configs_dir / "sample_config.yml"
    config.write_text(TEST_APP_CONFIG)

    parsed_config = fetch_config_from_yaml(config)

    # When
    with pytest.raises(ValidationError) as e:
        create_and_validate_config(parsed_config)

    # Then
    assert "field required" in str(e.value)
    assert "3 validation errors for AppConfig" in str(e.value)


def test_missing_config_model_fields_raises_error(tmpdir):

    # Given
    configs_dir = Path(tmpdir)
    config = configs_dir / "sample_config.yml"
    config.write_text(TEST_MODEL_CONFIG)

    parsed_config = fetch_config_from_yaml(config)

    # When
    with pytest.raises(ValidationError) as e:
        create_and_validate_config(parsed_config)

    # Then
    assert "field required" in str(e.value)
    assert "25 validation errors for ModelConfig" in str(e.value)
