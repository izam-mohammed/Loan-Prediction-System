from loanPrediction import logger
import pandas as pd
from loanPrediction.utils.common import is_nan
from loanPrediction.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.data = pd.read_csv(self.config.unzip_data_dir)

    def _validate_all_columns(self) -> bool:
        try:
            all_cols = list(self.data.columns)
            all_schema = self.config.all_schema.keys()

            validation_status = True
            for col in all_schema:
                if col not in all_cols:
                    validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def _validate_na_values(self) -> bool:
        try:
            total_rows = len(self.data)
            threshold = total_rows * self.config.nan_ratio
            all_cols = list(self.data.columns)
            na_values = self.data.isna().sum()

            validation_status = True
            for col in all_cols:
                if na_values[col] > threshold:
                    validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def _validate_categories(self) -> bool:
        try:
            categories = self.config.categories
            data = self.data

            validation_status = True
            for col in categories:
                for category in data[col].unique():
                    if category not in categories[col] and is_nan(category) == False:
                        validation_status = False

            return validation_status

        except Exception as e:
            raise e

    def final_validation(self) -> bool:
        try:
            validation_column = self._validate_all_columns()
            validation_na = self._validate_na_values()
            validation_categories = self._validate_categories()

            if validation_column and validation_na and validation_categories:
                validation_all = True
            else:
                validation_all = False

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(
                    f"Validation column status: {validation_column}\
                        \nValidation NA values status: {validation_na}\
                        \nValidation categorical columns: {validation_categories}\
                        \n\nValidation all: {validation_all}"
                )

        except Exception as e:
            raise e
