import pandas as pd
import logging

from houseregression_model.config.core import config
from houseregression_model.predict import make_predictions
from houseregression_model.processing.utility_functions import load_dataset

logging.basicConfig(level=logging.INFO)


def sample_data() -> pd.DataFrame:
    data = load_dataset(filename=config.app_config.training_data_file)
    training_data = data.copy()[-200:]
    return training_data


def create_prediction_data() -> None:
    data = sample_data()
    result = make_predictions(input_data=data)

    predictions = result.get("predictions")
    prediction_df = pd.DataFrame(data={"old_model_predictions": predictions})
    file_location = "housePriceModelling/houseregression_model"
    prediction_df.to_csv(f"{file_location}/old_predictions.csv", 
                         index=False,
                         sep=',',
                         encoding='utf-8')


create_prediction_data()
