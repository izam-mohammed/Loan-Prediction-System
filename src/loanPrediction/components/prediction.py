from loanPrediction import logger
from loanPrediction.utils.common import load_bin, save_json
import pandas as pd
from loanPrediction.entity.config_entity import PredictionConfig
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

class Prediction:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def predict(self):
        model = load_bin(Path(self.config.model_path))
        data = pd.read_csv(Path(self.config.data_path))

        categorical_cols = ["Gender",'Married','Education','Self_Employed','Property_Area']

        vectorizer = OrdinalEncoder()
        data[categorical_cols] = vectorizer.fit_transform(data[categorical_cols])

        prediction = model.predict(data)
        logger.info(f"predicted the new data as {prediction[0]}")


        save_json(path=self.config.prediction_file, data={'prediction':float(prediction[0])})