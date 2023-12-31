from loanPrediction.config.configuration import ConfigurationManager
from loanPrediction.components.data_validation import DataValiadtion
from loanPrediction import logger


STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        """
        Initialize DataValidationTrainingPipeline instance.
        """
        pass

    def main(self):
        """
        Execute the main steps of the data validation training pipeline.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.final_validation()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
