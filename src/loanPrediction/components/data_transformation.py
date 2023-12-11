import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from loanPrediction.utils.common import save_bin
from pathlib import Path
from sklearn.model_selection import train_test_split
from loanPrediction import logger
from loanPrediction.entity.config_entity import DataTransformationConfig
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        data = pd.read_csv(self.config.data_path)

        # take only needed cols:
        data = data[self.config.all_cols]

        # impute categorical data
        cols = data[
            [
                "Gender",
                "Married",
                "Self_Employed",
                "Education",
                "Credit_History",
                "Property_Area",
            ]
        ]
        for i in cols:
            data[i].fillna(data[i].mode().iloc[0], inplace=True)

        # impute the numerical data
        n_cols = data[
            ["LoanAmount", "Loan_Amount_Term", "CoapplicantIncome", "ApplicantIncome"]
        ]
        for i in n_cols:
            data[i].fillna(data[i].mean(axis=0), inplace=True)

        # drop rows don't have target
        if data.isna().any().any():
            data.dropna(inplace=True)

        # Transform data
        object_cols = [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]
        encoder = OrdinalEncoder()
        data[object_cols] = encoder.fit_transform(data[object_cols])
        data["Loan_Status"] = data["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)

        save_bin(
            data=encoder,
            path=Path(os.path.join(self.config.root_dir, self.config.encoder_name)),
        )
        data.to_csv(os.path.join(self.config.root_dir, "encoded_data.csv"), index=False)

    def split_data(self):
        try:
            data = pd.read_csv(os.path.join(self.config.root_dir, "encoded_data.csv"))
        except Exception as e:
            logger.log(f"can't get the encoded data encoded_data.csv")
            data = pd.read_csv(self.config.data_path)

        X = data.drop("Loan_Status", axis=1)
        y = data["Loan_Status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
