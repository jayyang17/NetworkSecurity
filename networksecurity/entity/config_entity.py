from datetime import datetime
import os
from networksecurity.constants import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp=timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name, timestamp)
        self.timestamps: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # data ingestion directory
        self.data_ingestion_dir: str=os.path.join(
            #artifact_dir is from training pipeline config, DATA_INGESTION_DIR_NAME is from constant.training_pipeline
            training_pipeline_config.artifact_dir, 
            training_pipeline.DATA_INGESTION_DIR_NAME
        )
        # feature store 
        self.feature_store_file_path: str=os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, 
            training_pipeline.FILE_NAME
        )
        # train file
        self.training_file_path: str=os.path.join(
            self.data_ingestion_dir, 
            training_pipeline.DATA_INGESTION_INGESTED_DIR, 
            training_pipeline.TRAIN_FILE_NAME
        )
        # test file
        self.testing_file_path: str=os.path.join(
            self.data_ingestion_dir, 
            training_pipeline.DATA_INGESTION_INGESTED_DIR, 
            training_pipeline.TEST_FILE_NAME
        )
        # train test split
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        # collection name
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        # database name
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME