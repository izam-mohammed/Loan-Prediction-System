from loanPrediction import logger
from loanPrediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from loanPrediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from loanPrediction.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

def run_pipeline(stage_name, pipeline_instance):
    """
    Run a specific stage of the sentiment analysis pipeline.

    Parameters:
    - stage_name: str
        Name of the pipeline stage.
    - pipeline_instance: object
        Instance of the pipeline stage to be executed.

    Returns:
        None
    """
    try:
        logger.info(f">>>>>> Stage {stage_name} started <<<<<<")
        pipeline_instance.main()
        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    
if __name__ == "__main__":
    run_pipeline("Data Ingestion", DataIngestionTrainingPipeline())
    run_pipeline("Data Validation", DataValidationTrainingPipeline())
    run_pipeline("Data Tranformation", DataTransformationTrainingPipeline())