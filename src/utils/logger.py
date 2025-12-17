import logging
import time
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split


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


