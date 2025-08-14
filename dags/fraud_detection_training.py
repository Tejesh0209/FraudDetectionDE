import os
import logging

from dotenv import load_dotenv
import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level = logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FraudDetectionTraning: 
    def __init__(self,config='/app/config.yaml'):
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet',
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'usr/bin/git'

        load_dotenv(dotenv_path='/app/.env')

        self.config = self._load_config(config_path)

        os.environ({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECERT_ACCESS_KEY': os.getenv('AWS_SECERT_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })

    def _load_config(self,config_path: str) -> dict:
        try:
            with open(config_path,'r') as f:
                config = yaml.safe_load(f)
            logger.error('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('failed to load configuration: %s', str(e))
            raise

