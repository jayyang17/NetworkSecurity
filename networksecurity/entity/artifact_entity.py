from dataclasses import dataclass

# output of the data ingestion component
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str