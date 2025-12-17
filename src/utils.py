import logging
import time

import mlflow
from mlflow.tracking import MlflowClient
import requests

def setup_common_logger():
    logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.Formatter.converter = time.gmtime
    


def verbose_log(msg: str):
    logging.info("\n" + "="*80)
    logging.info(msg)
    logging.info("="*80)




def check_mlflow_connection(uri):
    try:
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        client.search_experiments()
        return True
    except Exception as e:
        logging.error(f"MLflow connection failed: {e}")
        return False
