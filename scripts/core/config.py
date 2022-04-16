

from pydantic import BaseSettings


class MainConfig(BaseSettings):
    train_data_s3_path: str
    train_data_file_name: str
    mlflow_uri: str
    s3_uri: str
    aws_access_key_id: str
    aws_secret_access_key: str